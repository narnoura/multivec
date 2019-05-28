#pragma once
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <thread>
#include <assert.h>
#include <iomanip> // setprecision, setw, left
#include <chrono>
#include <iterator>
#include <unordered_map>
#include "vec.hpp"

using namespace std;
using namespace std::chrono;

const float MAX_EXP = 6;
const int UNIGRAM_TABLE_SIZE = 1e8; // size of the frequency table
//const string sentiment_labels[4] = {"positive", "negative", "neutral", "none"};
const string sentiment_labels[3] = {"positive", "negative", "neutral"};

typedef Vec vec;
typedef vector<vec> mat;

inline int sentimentIndex(const string& word, const unordered_map<string, string> & lexicon) {
    int index = 3;
    if (word.empty()) {
        std::cout << "sentimentIndex Warning: Empty word." << endl;
        return index;
    }
   
    unordered_map<string, string>::const_iterator found = lexicon.find(word); 
    // if (lexicon.find(word) != lexicon.end()) {
    if (found != lexicon.end()) {
        //std::cout << "found subjective word " << word << endl;
        string pol = found->second;
	//std::cout << "polarity is " << pol << endl;         
        if (pol == "positive") {
            index = 0;
        }
        else if (pol == "negative") {
            index = 1;
        }
        else if (pol == "neutral" or pol == "both") {
        // TODO can try ignoring
            index = 2;
        }
        else{
            std::cout << "Invalid value " << pol << endl;
        }
    }
   return index; 
        
}

inline string sentimentLabel(const int & sentimentIndex) {
    string label = "";
    if (sentimentIndex == 0) {
        label = "positive";
    }
    else if (sentimentIndex == 1) {
        label = "negative";
    }
    else if (sentimentIndex == 2) {
        label = "neutral";
    }
    else if (sentimentIndex == 3) {
        label = "none";
    }
    else{
        std::cout << "Invalid index " << sentimentIndex << endl;
    }
    return label;
    
}

inline float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

inline float softmax(const vec & vector, mat* matrix, int index, int N){
// This is actually softmax regression
    float x = vector.dot((*matrix)[index]);
    float num = exp(x);
    float sum = num;
    for (int i =0; i < N && i!=index; i++) {
        sum += exp(vector.dot((*matrix)[i]));
    }
    if (sum == 0) {
        if (num > 0) {
            cout << "How the hell?" << endl;
        }
        return 0.0;
    } else 
    return num / sum;

}

inline float cosineSimilarity(const vec &v1, const vec &v2) {
    if (v1.norm() == 0 || v2.norm() == 0) {
        cout << "Norm is 0, returning 0 " << endl;
	return 0.0;
    }
    else{
    	return v1.dot(v2) / (v1.norm() * v2.norm());
    }
}

inline string lower(string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

inline vector<string> split(const string& sequence) {
    vector<string> words;
    istringstream iss(sequence);
    string word;

    while (iss >> word) {
        words.push_back(word);
    }

    return words;
}

inline vector<string> split_tab(const string& sequence) {
    vector<string> fields;
    istringstream iss(sequence);
    string token;

    while (std::getline(iss, token, '\t')) {
        fields.push_back(token);
    }

    return fields;
}

inline void check_is_open(ifstream& infile, const string& filename) {
    if (!infile.is_open()) {
        throw runtime_error("couldn't open file " + filename);
    }
}

inline void check_is_open(ofstream& outfile, const string& filename) {
    if (!outfile.is_open()) {
        throw runtime_error("couldn't open file " + filename);
    }
}

inline void check_is_non_empty(ifstream& infile, const string& filename) {
    if (infile.peek() == std::ifstream::traits_type::eof()) {
        throw runtime_error("training file " + filename + " is empty");
    }
}

namespace multivec {
    /**
     * @brief Custom random generator. std::rand is thread-safe but very slow with multiple threads.
     * https://en.wikipedia.org/wiki/Linear_congruential_generator
     *
     * @return next random number
     */
    inline unsigned long long rand() {
        static unsigned long long next_random(time(NULL)); // in C++11 the thread_local keyword would solve the thread safety problem.
        next_random = next_random * static_cast<unsigned long long>(25214903917) + 11; // unsafe, but we don't care
        return next_random >> 16; // with this generator, the most significant bits are bits 47...16
    }

    inline float randf() {
        return  (multivec::rand() & 0xFFFF) / 65536.0f;
    }
}

/**
 * @brief Node of a Huffman binary tree, used for the hierarchical softmax algorithm.
 */
struct HuffmanNode {
    static const HuffmanNode UNK; // node for out-of-vocabulary words

    string word;

    vector<int> code; // Huffman code of this node: path from root to leaf (0 for left, 1 for right)
    vector<int> parents; // indices of the parent nodes

    HuffmanNode* left; // used for constructing the tree, useless afterwards
    HuffmanNode* right;

    int index;
    int count;

    bool is_leaf;
    bool is_unk;

    HuffmanNode() : index(-1), is_unk(true) {}

    HuffmanNode(int index, const string& word) :
            word(word), index(index), count(1), is_leaf(true), is_unk(false)
    {}

    HuffmanNode(int index, HuffmanNode* left, HuffmanNode* right) :
            left(left), right(right), index(index), count(left->count + right->count), is_leaf(false), is_unk(false)
    {}

    bool operator==(const HuffmanNode& node) const {
        return index == node.index;
    }

    bool operator!=(const HuffmanNode& node) const {
        return !(operator==(node));
    }

    static bool comp(const HuffmanNode* v1, const HuffmanNode* v2) {
        return (v1->count) > (v2->count);
    }
};

struct Config {
    float learning_rate;
    int dimension; // size of the embeddings
    int min_count; // minimum count of each word in the training file to be included in the vocabulary
    int iterations; // number of training epochs
    int window_size;
    int threads;
    float subsampling;
    bool verbose; // print additional information
    bool hierarchical_softmax;
    bool skip_gram; // set to true to use skip-gram model instead of CBOW
    int negative; // number of negative samples used for the negative sampling training algorithm
    bool sent_vector; // includes sentence vectors in the training
    bool learn_attn; //  learn attention weights for CBOW monlingual and bilingual models
    bool learn_sentiment; // learn sentiment output weights for (so far) CBOW bilingual models
    string sent_lexicon;  // path to read sentiment lexicon
    float gamma;

    Config() :
        learning_rate(0.05),
        //learning_rate(0.1),
	dimension(100),
        min_count(5),
        iterations(5),
        window_size(5),
        threads(4),
        subsampling(1e-03),
        verbose(false), // not serialized
        hierarchical_softmax(false),
        skip_gram(false),
        negative(5),
        sent_vector(false),
        learn_attn(false),
        learn_sentiment(false),
        sent_lexicon(""),
        gamma(1.0f) // it might be different for different monolingual models
        {}

    virtual void print() const {
        std::cout << std::boolalpha; // to print false/true instead of 0/1
        std::cout << "dimension:   " << dimension << std::endl;
        std::cout << "window size: " << window_size << std::endl;
        std::cout << "min count:   " << min_count << std::endl;
        std::cout << "alpha:       " << learning_rate << std::endl;
        std::cout << "iterations:  " << iterations << std::endl;
        std::cout << "threads:     " << threads << std::endl;
        std::cout << "subsampling: " << subsampling << std::endl;
        std::cout << "skip-gram:   " << skip_gram << std::endl;
        std::cout << "HS:          " << hierarchical_softmax << std::endl;
        std::cout << "negative:    " << negative << std::endl;
        std::cout << "sent vector: " << sent_vector << std::endl;
        std::cout << "learn attention: " << learn_attn << std::endl;
        std::cout << "learn sentiment: " << learn_sentiment << std::endl;
        std::cout << "sentiment lexicon:" << sent_lexicon << std::endl;
        std::cout << "gamma:" << gamma << std::endl;
    }
};

struct BilingualConfig : Config {
    float beta; 
    bool comparable; // assumes input is from comparable corpora, learns sentence alignments and word vectors
    BilingualConfig() : 
	beta(1.0f),
    	comparable(false)	
        {}
//  BilingualConfig() : beta(4.0f) {}
    void print() const {
        Config::print();
        std::cout << "beta:        " << beta << std::endl;
	std::cout << "comparable: " << comparable << std::endl;
    }
};

