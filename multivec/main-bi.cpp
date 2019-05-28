#include "bilingual.hpp"
#include <getopt.h>

struct option_plus {
    const char *name;
    int         has_arg;
    int        *flag;
    int         val;
    const char *desc;
};

static vector<option_plus> options_plus = {
    {"help",          no_argument,       0, 'h', "print this help message"},
    {"verbose",       no_argument,       0, 'v', "verbose mode"},
    {"dimension",     required_argument, 0, 'a', "dimension of the word embeddings"},
    {"min-count",     required_argument, 0, 'b', "minimum count of vocabulary words"},
    {"window-size",   required_argument, 0, 'c', "size of the window"},
    {"threads",       required_argument, 0, 'd', "number of threads"},
    {"iter",          required_argument, 0, 'e', "number of training epochs"},
    {"negative",      required_argument, 0, 'f', "number of negative samples (0 for no negative sampling)"},
    {"alpha",         required_argument, 0, 'g', "initial learning rate"},
    {"beta",          required_argument, 0, 'i', "bilingual training weight"},
    {"subsampling",   required_argument, 0, 'j', "subsampling (usually between 1e-03 and 1e-05)"},
    {"sg",            no_argument,       0, 'k', "skip-gram model (default: CBOW)"},
    {"hs",            no_argument,       0, 'l', "hierarchical softmax (default off)"},
    {"train-src",     required_argument, 0, 'm', "specify source file for training"},
    {"train-trg",     required_argument, 0, 'n', "specify target file for training"},
    {"load",          required_argument, 0, 'o', "load model"},
    {"save",          required_argument, 0, 'p', "save model"},
    {"save-src",      required_argument, 0, 'q', "save source model"},
    {"save-trg",      required_argument, 0, 'r', "save target model"},
    {"train-sentiment", required_argument,0,'y', "tsv lexicon for learning sentiment"},
    {"save-sentiment",  required_argument,0,'z', "where to save save sentiment weights"},
    {"gamma", required_argument,0,'t', "learning rate for sentiment"},
    {"sent-vector",   no_argument,0,'s',"learn bilingual sentence vectors"}, 
    {"attn",          no_argument,0,'w',"learn attention weights"},
    {"sentiment",     no_argument,0,'x',"learn sentiment vectors"},
    {"sentiment-policy", required_argument, 0, '$', "policy for saving sentiment vectors"},
  
    {0, 0, 0, 0, 0}
};

void print_usage() {
    std::cout << "Options:" << std::endl;
    for (auto it = options_plus.begin(); it != options_plus.end(); ++it) {
        if (it->name == 0) continue;
        string name(it->name);
        if (it->has_arg == required_argument) name += " arg";
        std::cout << std::setw(26) << std::left << "  --" + name << " " << it->desc << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char **argv) {

    vector<option> options;
    for (auto it = options_plus.begin(); it != options_plus.end(); ++it) {
        option op = {it->name, it->has_arg, it->flag, it->val};
        options.push_back(op);
    }

    string load_file;

    // first pass on parameters to find out if a model file is provided
    while (1) {
        int option_index = 0;
        int opt = getopt_long(argc, argv, "hv", options.data(), &option_index);
        if (opt == -1) break;

        switch (opt) {
            case 'o': load_file = string(optarg);           break;
            default:                                        break;
        }
    }

    BilingualConfig config;
    BilingualModel model(&config);

    // model file needs to be loaded before anything else (otherwise it overwrites the parameters)
    if (!load_file.empty()) {
        model.load(load_file);
    }

    string train_src_file;
    string train_trg_file;
    string save_file;
    string save_src_file;
    string save_trg_file;
    string train_sent_file;
    string save_sent_file;
    int sentiment_policy = 3;

    optind = 0;  // necessary to parse arguments twice
    while (1) {
        int option_index = 0;
        int opt = getopt_long(argc, argv, "hv", options.data(), &option_index);
        if (opt == -1) break;

        switch (opt) {
            case 0:                                         break;
            case 'h': print_usage();                        return 0;
            case 'v': config.verbose = true;                break;
            case 'a': config.dimension = atoi(optarg);      break;
            case 'b': config.min_count = atoi(optarg);      break;
            case 'c': config.window_size = atoi(optarg);    break;
            case 'd': config.threads = atoi(optarg);      break;
            case 'e': config.iterations = atoi(optarg); break;
            case 'f': config.negative = atoi(optarg);       break;
            case 'g': config.learning_rate = atof(optarg); break;
            case 'i': config.beta = atof(optarg);           break;
            case 'j': config.subsampling = atof(optarg);    break;
            case 'k': config.skip_gram = true;              break;
            case 'l': config.hierarchical_softmax = true;   break;
            case 'm': train_src_file = string(optarg);      break;
            case 'n': train_trg_file = string(optarg);      break;
            case 'o':                                       break;
            case 'p': save_file = string(optarg);           break;
            case 'q': save_src_file = string(optarg);       break;
            case 'r': save_trg_file = string(optarg);       break;
	    case 's': config.sent_vector = true;	    break;
//	    case 't': config.comparable = true;		    break;
	    case 'w': config.learn_attn = true;             break;
	    case 'x': config.learn_sentiment = true; break;
	    case 'y': train_sent_file = string(optarg);
            	     config.sent_lexicon = train_sent_file; break;
	    case 'z': save_sent_file = string(optarg); break;
	    case 't': config.gamma = atof(optarg); break;
	    case '$': sentiment_policy = atoi(optarg); break;
            default:                                        abort();
        }
    }

    if (load_file.empty() && (train_src_file.empty() || train_trg_file.empty())) {
        print_usage();
        return 0;
    }
    if (config.learn_sentiment && (train_sent_file.empty() || save_sent_file.empty())) {
        std::cout << "Specify sentiment lexicon file and sentiment output file" << endl;
        print_usage();
        return 0;
    }

    std::cout << "MultiVec-bi" << std::endl;
    config.print();

    if (!train_src_file.empty() && !train_trg_file.empty()) {
        
//	if (config.comparable) {
//	model.trainComparable(train_src_file, train_trg_file, load_file.empty());
//	} else{
	model.train(train_src_file, train_trg_file, load_file.empty());
//	}
    }

    if(!save_file.empty()) {
        model.save(save_file);
    }
    if(!save_src_file.empty()) {
        model.src_model.save(save_src_file);
    }
    if(!save_trg_file.empty()) {
        model.trg_model.save(save_trg_file);
    }
    if (config.learn_sentiment) {
        std::cout << "Saving sentiment vectors" << endl;
        model.saveSentimentVectors(save_sent_file, sentiment_policy);
    }

    return 0;
}
