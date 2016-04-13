#pragma once
#include "Import.hpp"
#include "ModelConfig.hpp"

#ifdef HDD_LOAD
const Dataset FB15K("FB15K", "D:\\Data\\Knowledge Embedding\\FB15K\\", "train.txt", "dev.txt", "test.txt", true);
const Dataset FB13("FB13", "D:\\Data\\Knowledge Embedding\\FB13\\", "train.txt", "dev.txt", "test.txt", false);
const Dataset WN11("WN11", "D:\\Data\\Knowledge Embedding\\WN11\\", "train.txt", "dev.txt", "test.txt", false);
const Dataset WN18("WN18", "D:\\Data\\Knowledge Embedding\\WN18\\", "train.txt", "dev.txt", "test.txt", true);
const Dataset Wordnet("Wordnet", "D:\\Data\\Knowledge Embedding\\Wordnet\\", "train.txt", "dev.txt", "test.txt", false);
const Dataset Freebase("Freebase", "D:\\Data\\Knowledge Embedding\\Freebase\\", "train.txt", "dev.txt", "test.txt", false);
const Dataset FB15KT("FB15KT", "D:\\Data\\Joint Knowledge and Text\\Release\\", "train.txt", "valid.txt", "test.txt", true);
const string report_path = "D:\\实验\\Report\\Experiment.Embedding\\";
const string semantic_vfile_FB15K = "D:\\Data\\Knowledge Embedding\\FB15K\\topics.bsd";
const string semantic_tfile_FB15K = "D:\\Data\\Knowledge Embedding\\FB15K\\description.txt";
const string semantic_vfile_WN18 = "D:\\Data\\Knowledge Embedding\\WN18\\topics.bsd";
const string semantic_tfile_WN18 = "D:\\Data\\Knowledge Embedding\\WN18\\descriptions.txt";
const string semantic_vfile_FB15KZS = "D:\\Data\\Knowledge Embedding\\FB15KZS\\topics.bsd";
const string semantic_tfile_FB15KZS = "D:\\Data\\Knowledge Embedding\\FB15KZS\\description.txt";
const string type_file_FB15K = "D:\\Data\\Knowledge Embedding\\FB15K\\type.txt";
const string type_file_FB15KZS = "D:\\Data\\Knowledge Embedding\\FB15K\\type.txt";
const string triple_zeroshot_FB15K = "D:\\Data\\Knowledge Embedding\\FB15KZS\\zeroshot.txt";
#endif

#ifdef SSD_LOAD
const Dataset FB15K("FB15K", "C:\\Data\\Knowledge Embedding\\FB15K\\", "train.txt", "dev.txt", "test.txt", true);
const Dataset FB13("FB13", "C:\\Data\\Knowledge Embedding\\FB13\\", "train.txt", "dev.txt", "test.txt", false);
const Dataset WN11("WN11", "C:\\Data\\Knowledge Embedding\\WN11\\", "train.txt", "dev.txt", "test.txt", false);
const Dataset WN18("WN18", "C:\\Data\\Knowledge Embedding\\WN18\\", "train.txt", "dev.txt", "test.txt", true);
const Dataset Wordnet("Wordnet", "C:\\Data\\Knowledge Embedding\\Wordnet\\", "train.txt", "dev.txt", "test.txt", false);
const Dataset Freebase("Freebase", "C:\\Data\\Knowledge Embedding\\Freebase\\", "train.txt", "dev.txt", "test.txt", false);
const Dataset FB15KT("FB15KT", "C:\\Data\\Joint Knowledge and Text\\Release\\", "train.txt", "valid.txt", "test.txt", true);
const string report_path = "D:\\实验\\Report\\Experiment.Embedding\\";
const string semantic_vfile_FB15K = "C:\\Data\\Knowledge Embedding\\FB15K\\topics.bsd";
const string semantic_tfile_FB15K = "C:\\Data\\Knowledge Embedding\\FB15K\\description.txt";
const string semantic_vfile_WN18 = "C:\\Data\\Knowledge Embedding\\WN18\\topics.bsd";
const string semantic_tfile_WN18 = "C:\\Data\\Knowledge Embedding\\WN18\\descriptions.txt";
const string type_file_FB15K = "C:\\Data\\Knowledge Embedding\\FB15K\\type.txt";
const string type_file_FB15KZS = "C:\\Data\\Knowledge Embedding\\FB15KZS\\type.txt";
const string triple_zeroshot_FB15K = "C:\\Data\\Knowledge Embedding\\FB15KZS\\zeroshot.txt";
const string semantic_vfile_FB15KZS = "C:\\Data\\Knowledge Embedding\\FB15KZS\\topics.bsd";
const string semantic_tfile_FB15KZS = "C:\\Data\\Knowledge Embedding\\FB15KZS\\description.txt";
#endif

#ifdef LINUX_LOAD
const Dataset FB15K("FB15K", "/home/bookman/data/Knowledge Embedding/FB15K/", "train.txt", "dev.txt", "test.txt", true);
const Dataset FB13("FB13", "/home/bookman/data/Knowledge Embedding/FB13/", "train.txt", "dev.txt", "test.txt", false);
const Dataset WN11("WN11", "/home/bookman/data/Knowledge Embedding/WN11/", "train.txt", "dev.txt", "test.txt", false);
const Dataset WN18("WN18", "/home/bookman/data/Knowledge Embedding/WN18/", "train.txt", "dev.txt", "test.txt", true);
const Dataset Wordnet("Wordnet", "/home/bookman/data/Knowledge Embedding/Wordnet/", "train.txt", "dev.txt", "test.txt", false);
const Dataset Freebase("Freebase", "/home/bookman/data/Knowledge Embedding/Freebase/", "train.txt", "dev.txt", "test.txt", false);
const string report_path = "/home/bookman/Report/";
const string semantic_vfile_FB15K = "/home/bookman/data/Knowledge Embedding/FB15K/topics.bsd";
const string semantic_tfile_FB15K = "/home/bookman/data/Knowledge Embedding/FB15K/description.txt";
const string semantic_vfile_WN18 = "/home/bookman/data/Knowledge Embedding/WN18/topics.bsd";
const string semantic_tfile_WN18 = "/home/bookman/data/Knowledge Embedding/WN18/descriptions.txt";
const string type_file_FB15K = "/home/bookman/data/Knowledge Embedding/FB15K/type.txt";
const string type_file_FB15KZS = "/home/bookman/data/Knowledge Embedding/FB15KZS/type.txt";
const string triple_zeroshot_FB15K = "/home/bookman/data/Knowledge Embedding/FB15KZS/zeroshot.txt";
const string semantic_vfile_FB15KZS = "/home/bookman/data/Knowledge Embedding/FB15KZS/topics.bsd";
const string semantic_tfile_FB15KZS = "/home/bookman/data/Knowledge Embedding/FB15KZS/description.txt";
#endif