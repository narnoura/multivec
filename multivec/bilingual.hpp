#pragma once
#include "monolingual.hpp"
#include "utils.hpp"

using namespace std;

class BilingualModel
{
    friend void save(ofstream& outfile, const BilingualModel& model);
    friend void load(ifstream& infile, BilingualModel& model);

private:
    // Configuration of the model (monolingual models have the same configuration)
    BilingualConfig* const config;

    long long words_processed; // number of words processed so far
    float alpha;

    void trainChunk(const string& src_file,
                    const string& trg_file,
                    const vector<long long>& src_chunks,
                    const vector<long long>& trg_chunks,
                    int thread_id);

    // TODO: unsupervised alignment (GIZA)
    vector<int> uniformAlignment(const vector<HuffmanNode>& src_nodes, const vector<HuffmanNode>& trg_nodes);

    int trainSentence(const string& trg_sent, const string& src_sent);

    void trainWord(MonolingualModel& src_params, MonolingualModel& trg_params,
        const vector<HuffmanNode>& src_nodes, const vector<HuffmanNode>& trg_nodes,
        int src_pos, int trg_pos, float alpha, float gamma, mat* k, vec* s, 
        bool has_lexicon=true);

    void trainWordCBOW(MonolingualModel&, MonolingualModel&,
        const vector<HuffmanNode>&, const vector<HuffmanNode>&,
        int, int, float);

    void trainWordCBOWAttn(MonolingualModel&, MonolingualModel&,
        const vector<HuffmanNode>&, const vector<HuffmanNode>&,
        int, int, float, mat* k, vec* s);
        
    void trainWordCBOWSentiment(MonolingualModel&, MonolingualModel&,
        const vector<HuffmanNode>&, const vector<HuffmanNode>&,
        int, int, float, float, bool);

    void trainWordSkipGram(MonolingualModel&, MonolingualModel&,
        const vector<HuffmanNode>&, const vector<HuffmanNode>&,
        int, int, float);
        

public:
    // A bilingual model is comprised of two monolingual models
    MonolingualModel src_model;
    MonolingualModel trg_model;
	
    // Attention matrices 
    mat src_trg_attn;
    vec src_trg_bias; //si 
    mat trg_src_attn;
    vec trg_src_bias;
    
    // Sentiment weights for bilingual model
    mat sentiment_weights; // sentiment weights for bilingual training
    //mat in_sentiment_weights; // input sentiment weights for bilingual training
	
    // prefer this constructor
    BilingualModel(BilingualConfig* config) : config(config), src_model(config), trg_model(config) {}

    void train(const string& src_file, const string& trg_file, bool initialize = true);
    void load(const string& filename);
    void save(const string& filename) const;
    void saveSentimentVectors(const string &filename, int policy = 3) const;

    void initAttention();
    void initSentiment();

    float similarity(const string& src_word, const string& trg_word, int policy = 0) const; // cosine similarity
    float distance(const string& src_word, const string& trg_word, int policy = 0) const; // 1 - cosine similarity
    float similarityNgrams(const string& src_seq, const string& trg_seq, int policy = 0) const; // similarity between two sequences of same size
    float similaritySentence(const string& src_seq, const string& trg_seq, int policy = 0) const; // similarity between two variable-size sequences
    // similarity between two variable-size sequences taking into account part-of-speech tags and inverse document frequencies of terms in the sequences
    float similaritySentenceSyntax(const string& src_seq, const string& trg_seq, const string& src_tags, const string& trg_tags,
                                   const vector<float>& src_idf, const vector<float>& trg_idf, float alpha = 0.0, int policy = 0) const;
    
    vector<pair<string, float>> trg_closest(const string& src_word, int n = 10, int policy = 0) const; // n closest words to given word
    vector<pair<string, float>> src_closest(const string& trg_word, int n = 10, int policy = 0) const;
 
    mat attention_matrix (bool src_to_target, bool monolingual) const;
    vec bias_vector (bool src_to_target, bool monolingual) const;

};
