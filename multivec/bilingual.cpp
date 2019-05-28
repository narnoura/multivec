#include "bilingual.hpp"
#include "serialization.hpp"

void BilingualModel::initAttention() {

std::cout<< "Initializing attention parameters in bilingual model" << std::endl;
int v_src = static_cast<int>(src_model.vocabulary.size());
int v_trg = static_cast<int>(trg_model.vocabulary.size());
int w = config->window_size;

src_trg_attn = mat(v_src,vec(2*w));
trg_src_attn = mat(v_trg,vec(2*w));
src_trg_bias = vec(2*w);
trg_src_bias = vec(2*w);

for (size_t row = 0; row < v_src; ++row) {
      for (size_t col = 0; col< 2*w ; ++ col) {
        src_trg_attn[row][col]=0;
	 }
}
for (size_t row = 0; row < v_trg; ++row) {
      for (size_t col = 0; col< 2*w ; ++ col) {
	trg_src_attn[row][col]=0;
       }
}

for (size_t s =0; s < 2*w; ++ s) {
   src_trg_bias[s]=0;
   trg_src_bias[s]=0;
}

}

void BilingualModel::initSentiment() {

// Sentiment indices will correspond to the positions defined in sentiment_labels
// in utils.hpp
std::cout<< "Initializing sentiment output parameters in bilingual model" << std::endl;
int d = config->dimension;
int s = sizeof(sentiment_labels)/sizeof(sentiment_labels[0]);
std::cout<< "Number of sentiment labels: " << s << std::endl;
sentiment_weights = mat(s, vec(d));
//in_sentiment_weights = mat(s, vec(d));
for (size_t row = 0; row < s; ++row) {
   for (size_t col = 0; col < d; ++col) {
         //(was initializing the output sentiment weights before)
        sentiment_weights[row][col] = (multivec::randf() - 0.5f) / d;
        //in_sentiment_weights[row][col] = (multivec::randf() - 0.5f) / d;
    }
 }

}

void BilingualModel::train(const string& src_file, const string& trg_file, bool initialize) {
    std::cout << "Training files: " << src_file << ", " << trg_file << std::endl;

    if (initialize) {
        if (config->verbose)
            std::cout << "Creating new model" << std::endl;

        src_model.readVocab(src_file);
        trg_model.readVocab(trg_file);
        src_model.initNet();
        trg_model.initNet();
	if (config->learn_attn) {
	    initAttention();
	}
	if (config->learn_sentiment) {
        	src_model.readLexicon(config->sent_lexicon);
        	initSentiment();
	}
    } else {
        // TODO: check that initialization is fine
    }

    words_processed = 0;
    alpha = config->learning_rate;

    // read files to find out the beginning of each chunk
    auto src_chunks = src_model.chunkify(src_file, config->threads);
    auto trg_chunks = trg_model.chunkify(trg_file, config->threads);

    high_resolution_clock::time_point start = high_resolution_clock::now();
    if (config->threads == 1) {
        trainChunk(src_file, trg_file, src_chunks, trg_chunks, 0);
    } else {
        vector<thread> threads;

        for (int i = 0; i < config->threads; ++i) {
            threads.push_back(thread(&BilingualModel::trainChunk, this,
                src_file, trg_file, src_chunks, trg_chunks, i));
        }

        for (auto it = threads.begin(); it != threads.end(); ++it) {
            it->join();
        }
    }
    high_resolution_clock::time_point end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();

    if (config->verbose)
        std::cout << std::endl;

    std::cout << "Training time: " << static_cast<float>(duration) / 1000000 << std::endl;
}

void BilingualModel::trainChunk(const string& src_file,
                                const string& trg_file,
                                const vector<long long>& src_chunks,
                                const vector<long long>& trg_chunks,
                                int chunk_id) {
    ifstream src_infile(src_file);
    ifstream trg_infile(trg_file);

    try {
        check_is_open(src_infile, src_file);
        check_is_open(trg_infile, trg_file);
        check_is_non_empty(src_infile, src_file);
        check_is_non_empty(trg_infile, trg_file);
    } catch (...) {
        throw;
    }
    
    float starting_alpha = config->learning_rate;
    int max_iterations = config->iterations;
    long long training_words = src_model.training_words + trg_model.training_words;

    for (int k = 0; k < max_iterations; ++k) {
        int word_count = 0, last_count = 0;

        src_infile.clear();
        trg_infile.clear();
        src_infile.seekg(src_chunks[chunk_id], src_infile.beg);
        trg_infile.seekg(trg_chunks[chunk_id], trg_infile.beg);

        string src_sent, trg_sent;
        while (getline(src_infile, src_sent) && getline(trg_infile, trg_sent)) {
            word_count += trainSentence(src_sent, trg_sent);

            // update learning rate
            if (word_count - last_count > 10000) {
                words_processed += word_count - last_count; // asynchronous update
                last_count = word_count;

                alpha = starting_alpha * (1 - static_cast<float>(words_processed) / (max_iterations * training_words));
                alpha = std::max(alpha, starting_alpha * 0.0001f);

                if (config->verbose) {
                    printf("\rAlpha: %f  Progress: %.2f%%", alpha, 100.0 * words_processed /
                                    (max_iterations * training_words));
                    fflush(stdout);
                }
            }

            // stop when reaching the end of a chunk
            if (chunk_id < src_chunks.size() - 1 && src_infile.tellg() >= src_chunks[chunk_id + 1])
                break;
        }

        words_processed += word_count - last_count;
    }
}

vector<int> BilingualModel::uniformAlignment(const vector<HuffmanNode>& src_nodes,
                                             const vector<HuffmanNode>& trg_nodes) {
    vector<int> alignment; // index = position in src_nodes, value = position in trg_nodes (or -1)

    vector<int> trg_mapping; // maps positions in trg_sent to positions in trg_nodes (or -1)
    int k = 0;
    for (auto it = trg_nodes.begin(); it != trg_nodes.end(); ++it) {
        trg_mapping.push_back(*it == HuffmanNode::UNK ? -1 : k++);
    }

    for (int i = 0; i < src_nodes.size(); ++i) {
        int j = i * trg_nodes.size() / src_nodes.size();

        if (src_nodes[i] != HuffmanNode::UNK) {
            alignment.push_back(trg_mapping[j]);
        }
    }

    return alignment;
}

int BilingualModel::trainSentence(const string& src_sent, const string& trg_sent) {
    auto src_nodes = src_model.getNodes(src_sent);  // same size as src_sent, OOV words are replaced by <UNK>
    auto trg_nodes = trg_model.getNodes(trg_sent);

    // counts the number of words that are in the vocabulary
    int words = 0;
    words += src_nodes.size() - count(src_nodes.begin(), src_nodes.end(), HuffmanNode::UNK);
    words += trg_nodes.size() - count(trg_nodes.begin(), trg_nodes.end(), HuffmanNode::UNK);

    if (config->subsampling > 0) {
        src_model.subsample(src_nodes); // puts <UNK> tokens in place of the discarded tokens
        trg_model.subsample(trg_nodes);
    }

    if (src_nodes.empty() || trg_nodes.empty()) {
        return words;
    }

    // The <UNK> tokens are necessary to perform the alignment (the nodes vector should have the same size
    // as the original sentence)
    auto alignment = uniformAlignment(src_nodes, trg_nodes);

    // remove <UNK> tokens
    src_nodes.erase(
        std::remove(src_nodes.begin(), src_nodes.end(), HuffmanNode::UNK),
        src_nodes.end());
    trg_nodes.erase(
        std::remove(trg_nodes.begin(), trg_nodes.end(), HuffmanNode::UNK),
        trg_nodes.end());
        
    bool has_lexicon = false;

    // Monolingual training
    // TODO add sentiment attention weights to this function and call with pointer
    // may not need to have the extra parameter
    for (int src_pos = 0; src_pos < src_nodes.size(); ++src_pos) {
	// trainWord(src_model, src_model, src_nodes, src_nodes, src_pos, src_pos, alpha,true,false);
	trainWord(src_model, src_model, src_nodes, src_nodes, src_pos, src_pos, alpha, config->gamma, &src_model.attn_weights, &src_model.attn_bias);
    }
    //std::cout<< "Done with monolingual source" << std::endl;

    for (int trg_pos = 0; trg_pos < trg_nodes.size(); ++trg_pos) {
        //trainWord(trg_model, trg_model, trg_nodes, trg_nodes, trg_pos, trg_pos, alpha,true,false);
	//std::cout << "k[trg_nodes[pos].index][0]: " << trg_model.attn_weights[trg_nodes[trg_pos].index][0] << std::endl;
        trainWord(trg_model, trg_model, trg_nodes, trg_nodes, trg_pos, trg_pos, alpha, 0.0, &trg_model.attn_weights, &trg_model.attn_bias, has_lexicon);
    }

    //std::cout<< "Done with monolingual target" << std::endl;

    if (config->beta == 0)
        return words;

    // Bilingual training
    // For sentiment training, we can set gamma to 0 for models that predict the target word (since we don't know its sentiment)
    // If we set it to value of gamma, we give it the same sentiment as the source word it aligns to
    
    for (int src_pos = 0; src_pos < src_nodes.size(); ++src_pos) {
        // 1-1 mapping between src_nodes and trg_nodes
        int trg_pos = alignment[src_pos];

        if (trg_pos != -1) { // target word isn't OOV
            //trainWord(src_model, trg_model, src_nodes, trg_nodes, src_pos, trg_pos, alpha * config->beta,false,true);
        	    trainWord(src_model, trg_model, src_nodes, trg_nodes, src_pos, trg_pos, alpha * config->beta, config->gamma, &trg_src_attn, &trg_src_bias);
        	    // std::cout<< "Done with training src_target word (predicts src word) " << std::endl;
             trainWord(trg_model, src_model, trg_nodes, src_nodes, trg_pos, src_pos, alpha * config->beta, 0.0, &src_trg_attn, &src_trg_bias, has_lexicon);
        	   // trainWord(trg_model, src_model, trg_nodes, src_nodes, trg_pos, src_pos, alpha * config->beta, config->gamma, &src_trg_attn, &src_trg_bias, has_lexicon);
        	    //std::cout<< "Done with training target_src word (predicts trg_word)" << std::endl;
        }
    }
    //std::cout<< "Done with bilingual " << std::endl;
    //std::cout << "Done with sentence" << std::endl;
    return words; // returns the number of words processed (for progress estimation)
}

void BilingualModel::trainWord(MonolingualModel& src_model, MonolingualModel& trg_model,
                               const vector<HuffmanNode>& src_nodes, const vector<HuffmanNode>& trg_nodes,
                               int src_pos, int trg_pos, float alpha, float gamma, mat* k, vec* s, bool has_lexicon) {

    if (config->skip_gram) {
        return trainWordSkipGram(src_model, trg_model, src_nodes, trg_nodes, src_pos, trg_pos, alpha);
    } else {

	if (config->learn_attn) {
		//std::cout<< "Training word with CBOW Attn" << std::endl;
		return trainWordCBOWAttn(src_model, trg_model, src_nodes, trg_nodes, src_pos, trg_pos, alpha, k, s);
	} 
	else {
        	
        	if (config->learn_sentiment) {
        	
            	//std::cout << "Training word with sentiment" << std::endl;
            	return trainWordCBOWSentiment(src_model, trg_model, src_nodes, trg_nodes, src_pos, trg_pos, alpha, gamma, has_lexicon);
        	}
		
		else return trainWordCBOW(src_model, trg_model, src_nodes, trg_nodes, src_pos, trg_pos, alpha);
		// TODO add the sentiment if statement and call here
        	}
    }
}

void BilingualModel::trainWordCBOW(MonolingualModel& src_model, MonolingualModel& trg_model,
                                   const vector<HuffmanNode>& src_nodes, const vector<HuffmanNode>& trg_nodes,
                                   int src_pos, int trg_pos, float alpha) {
    // Trains the model by predicting a source node from its aligned context in the target sentence.
    // This function can be used in the reverse direction just by reversing the arguments. Likewise,
    // for monolingual training, use the same values for source and target.

    // 'src_pos' is the position in the source sentence of the current node to predict
    // 'trg_pos' is the position of the corresponding node in the target sentence

    int dimension = config->dimension;
    vec hidden(dimension, 0);
    HuffmanNode cur_node = src_nodes[src_pos];

    int this_window_size = 1 + multivec::rand() % config->window_size;
    //int this_window_size = config->window_size;
    int count = 0;

    float numerator = 0;	

    for (int pos = trg_pos - this_window_size; pos <= trg_pos + this_window_size; ++pos) {
        if (pos < 0 || pos >= trg_nodes.size() || pos == trg_pos) continue;
	hidden += trg_model.input_weights[trg_nodes[pos].index];
        ++count;
    }

    if (count == 0) return;
    hidden /= count;

    vec error(dimension, 0); // compute error & update output weights
    if (config->hierarchical_softmax) {
        error += src_model.hierarchicalUpdate(cur_node, hidden, alpha);
    }
    if (config->negative > 0) {
        error += src_model.negSamplingUpdate(cur_node, hidden, alpha);
    }

    // Update input weights
    for (int pos = trg_pos - this_window_size; pos <= trg_pos + this_window_size; ++pos) {
        if (pos < 0 || pos >= trg_nodes.size() || pos == trg_pos) continue;
        trg_model.input_weights[trg_nodes[pos].index] += error;
    } 
}

void BilingualModel::trainWordCBOWSentiment(MonolingualModel& src_model, MonolingualModel& trg_model,
                                   const vector<HuffmanNode>& src_nodes, const vector<HuffmanNode>& trg_nodes,
                                   int src_pos, int trg_pos, float alpha, float gamma, bool has_lexicon = true) {
    // Trains the model by predicting a source node from its aligned context in the target sentence.
    // This function can be used in the reverse direction just by reversing the arguments. Likewise,
    // for monolingual training, use the same values for source and target.
    
    // Learns sentiment vectors that predict the probability of the center node having prior sentiment polarity
    // Only sentiment vectors for subjective labels (i.e positive, negative, neutral) will be sampled and updated

    // 'src_pos' is the position in the source sentence of the current node to predict
    // 'trg_pos' is the position of the corresponding node in the target sentence

    int dimension = config->dimension;
    vec hidden(dimension, 0);
    HuffmanNode cur_node = src_nodes[src_pos];
    string word;
    int sentiment_index = 3;
    int in_sentiment_index = 3;
    
    if (has_lexicon) {
       word = cur_node.word;
       sentiment_index = sentimentIndex(word, src_model.sentiment_lexicon);
    } 
    else if (gamma != 0.0) {
    // Can do it even if gamma is 0
       word = trg_nodes[trg_pos].word; // use the sentiment lexicon from the other language
       sentiment_index = sentimentIndex(word, trg_model.sentiment_lexicon);
    }

    int this_window_size = 1 + multivec::rand() % config->window_size;
    //int this_window_size = config->window_size;
    int count = 0;

    float numerator = 0;	

    for (int pos = trg_pos - this_window_size; pos <= trg_pos + this_window_size; ++pos) {
        if (pos < 0 || pos >= trg_nodes.size() || pos == trg_pos) continue;
	hidden += trg_model.input_weights[trg_nodes[pos].index];
	
	// if updating input sentiment weights also
	/*in_sentiment_index = sentimentIndex(trg_nodes[pos].word, src_model.sentiment_lexicon);
	if (in_sentiment_index <=2 ){
        hidden += in_sentiment_weights[in_sentiment_index];
	}*/
	
        ++count;
    }

    if (count == 0) return;
    hidden /= count;

    vec error(dimension, 0); // compute error & update output weights
    vec sentiment_error(dimension, 0);
    /*if (config->hierarchical_softmax) {
        error += src_model.hierarchicalUpdate(cur_node, hidden, alpha);
    }*/
    if (config->negative > 0) {
        error += src_model.negSamplingUpdate(cur_node, hidden, alpha);
        
       // if ((has_lexicon || gamma != 0.0) && sentiment_index <=2) 
       if (has_lexicon && sentiment_index <= 2) {
            //sentiment_error += src_model.negSamplingUpdateSentimentSoftmax(cur_node, sentiment_index, hidden, &sentiment_weights, alpha, gamma);
            sentiment_error += src_model.negSamplingUpdateSentiment(cur_node, sentiment_index, hidden, &sentiment_weights, alpha, gamma);
        } else if (gamma != 0.0 && sentiment_index <=2 ) {
            // sample the sentiment error using the source language
            sentiment_error += trg_model.negSamplingUpdateSentiment(trg_nodes[trg_pos], sentiment_index, hidden, &sentiment_weights, alpha, gamma);
        } 
    }

    // Update input weights
    for (int pos = trg_pos - this_window_size; pos <= trg_pos + this_window_size; ++pos) {
        if (pos < 0 || pos >= trg_nodes.size() || pos == trg_pos) continue;
        trg_model.input_weights[trg_nodes[pos].index] += error;
        
        // if updating input sentiment weights also
        /*in_sentiment_index = sentimentIndex(trg_nodes[pos].word, src_model.sentiment_lexicon);
        if (in_sentiment_index <= 2) {
            in_sentiment_weights[in_sentiment_index] += error;
        } */
        
        if ((has_lexicon || gamma != 0.0) && sentiment_index <= 2) {
            trg_model.input_weights[trg_nodes[pos].index] += sentiment_error;
            
            // if updating input sentiment weights also
           /* if (in_sentiment_index <= 2) {
                in_sentiment_weights[in_sentiment_index] += sentiment_error;
            } */
        }
    } 
}

void BilingualModel::trainWordCBOWAttn(MonolingualModel& src_model, MonolingualModel& trg_model,
                                   const vector<HuffmanNode>& src_nodes, const vector<HuffmanNode>& trg_nodes,
                                   int src_pos, int trg_pos, float alpha, mat* k, vec* s) {
    // Trains the model by predicting a source node from its aligned context in the target sentence.
    // This function can be used in the reverse direction just by reversing the arguments. Likewise,
    // for monolingual training, use the same values for source and target.

    // 'src_pos' is the position in the source sentence of the current node to predict
    // 'trg_pos' is the position of the corresponding node in the target sentence
	
    //  Uses attention weights to prioritize contexts from the window
    
    int dimension = config->dimension;
    vec hidden(dimension, 0);
    HuffmanNode cur_node = src_nodes[src_pos];

    int this_window_size = 1 + multivec::rand() % config->window_size;
    int max_window_size = config->window_size;
    int count = 0;
    float denominator = 0;
    vec a(2*max_window_size);
    for (int c =0;c<2*max_window_size;c++) {
 	a[c]=0;	
    }
     
    int i = 0;
    for (int pos = trg_pos - max_window_size; pos <= trg_pos + max_window_size; ++pos) {
	
	if (pos < 0 || pos >= trg_nodes.size() || pos == trg_pos 
		|| abs(trg_pos - pos) > this_window_size) { 
		if (pos!=trg_pos) {
			i+=1;
		}
		continue;
	}
	++ count;
	denominator += exp((*k)[trg_nodes[pos].index][i] + (*s)[i]);
	/*if (i > 2*max_window_size - 1) {
	std::cout << "i is bigger than context window!" << std::endl;
	std::cout << "i: " << i << std::endl;
	return;	
	}*/
       i+=1;
     }

    int j=0;
    for (int pos = trg_pos - max_window_size; pos <= trg_pos + max_window_size; ++pos) {

	if (pos < 0 || pos >= trg_nodes.size() || pos == trg_pos
	     ||	abs(trg_pos - pos) > this_window_size) {
		if (pos!=trg_pos) { j+=1; }
		continue;
	 	}
	a[j] = exp((*k)[trg_nodes[pos].index][j] + (*s)[j]) / denominator;
	hidden += a[j]*trg_model.input_weights[trg_nodes[pos].index];
	j+=1;
     }

    if (count == 0) return;
    //hidden /= count;

   vec error(dimension, 0); // compute error & update output weights
   /* if (config->hierarchical_softmax) {
	 error += src_model.hierarchicalUpdateAttn(cur_node, trg_nodes, trg_model, trg_pos,hidden, alpha, a, k,s);
  }*/

   if (config->negative > 0) {
        error += src_model.negSamplingUpdateAttn(cur_node, trg_nodes, trg_model,trg_pos, this_window_size, hidden, alpha, a, k,s);
   }  

    // Update input weights
    int p = 0;
    for (int pos = trg_pos - max_window_size; pos <= trg_pos + max_window_size; ++pos) {
        if (pos < 0 || pos >= trg_nodes.size() || pos == trg_pos
		|| abs(trg_pos - pos) > this_window_size) {
		if (pos!=trg_pos) {p+=1;}
		 continue;
	}
	trg_model.input_weights[trg_nodes[pos].index] += error;
	p+=1;
    }
}

void BilingualModel::trainWordSkipGram(MonolingualModel& src_model, MonolingualModel& trg_model,
                                       const vector<HuffmanNode>& src_nodes, const vector<HuffmanNode>& trg_nodes,
                                       int src_pos, int trg_pos, float alpha) {
    HuffmanNode input_word = src_nodes[src_pos];

    int this_window_size = 1 + multivec::rand() % config->window_size;

    for (int pos = trg_pos - this_window_size; pos <= trg_pos + this_window_size; ++pos) {
        if (pos < 0 || pos >= trg_nodes.size() || pos == trg_pos) continue;
        HuffmanNode output_word = trg_nodes[pos];
        
        vec error(config->dimension, 0);
        if (config->hierarchical_softmax) {
            error += trg_model.hierarchicalUpdate(output_word, src_model.input_weights[input_word.index], alpha);
        }
        if (config->negative > 0) {
            error += trg_model.negSamplingUpdate(output_word, src_model.input_weights[input_word.index], alpha);
        }

        src_model.input_weights[input_word.index] += error;
    }
}

void BilingualModel::load(const string& filename) {
    if (config->verbose)
        std::cout << "Loading model" << std::endl;

    ifstream infile(filename);

    try {
        check_is_open(infile, filename);
    } catch (...) {
        throw;
    }

    ::load(infile, *this);
    src_model.initUnigramTable();
    trg_model.initUnigramTable();
}

void BilingualModel::save(const string& filename) const {
    if (config->verbose)
        std::cout << "Saving model" << std::endl;

    ofstream outfile(filename);

    try {
        check_is_open(outfile, filename);
    } catch (...) {
        throw;
    }

    ::save(outfile, *this);
}

void BilingualModel::saveSentimentVectors(const string &filename, int policy) const {
    if (config->verbose)
        std::cout << "Saving sentiment vectors in text format to " << filename << std::endl;

    ofstream outfile(filename, ios::binary | ios::out);

    try {
        check_is_open(outfile, filename);
    } catch (...) {
        throw;
    }

    int index = 0;
    int d = config->dimension;
    int s = sizeof(sentiment_labels)/sizeof(sentiment_labels[0]);
    
    outfile << s << " " << d << endl;
    
    for (auto it = sentiment_weights.begin(); it != sentiment_weights.end(); ++it) {
        /*  for (auto it = vocabulary.begin(); it != vocabulary.end(); ++it) {
        outfile << it->second.word << " ";
        vec embedding = wordVec(it->second.index, policy); } */
        std::cout << "index: " << index << " ";
        string label = sentimentLabel(index);
        std::cout << "label: " << label << endl;
        outfile << label << " ";
        vec embedding = *it; // by default, sentiment output weights
        /*if (policy == 0){
            // only input sentiment weights
            embedding = in_sentiment_weights[index];
        }
        else if (policy == 2) {
            // sum
            embedding += in_sentiment_weights[index];
        }*/
        for (int c = 0; c < config->dimension; ++c) {
            outfile << embedding[c] << " ";
            //std::cout << embedding[c] << " ";
        }
        outfile << endl;
        index +=1;
    }
}
