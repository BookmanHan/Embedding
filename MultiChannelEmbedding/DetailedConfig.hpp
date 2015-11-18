#pragma once
#include "Import.hpp"
#include "ModelConfig.hpp"

const Dataset FB15K("FB15K", "D:\\Data\\Knowledge Embedding\\FB15K\\", "train.txt", "dev.txt", "test.txt", true);
const Dataset FB13("FB13", "D:\\Data\\Knowledge Embedding\\FB13\\", "train.txt", "dev.txt", "test.txt", false);
const Dataset WN11("WN11", "D:\\Data\\Knowledge Embedding\\WN11\\", "train.txt", "dev.txt", "test.txt", false);
const Dataset WN18("WN18", "D:\\Data\\Knowledge Embedding\\WN18\\", "train.txt", "dev.txt", "test.txt", true);
const Dataset Wordnet("Wordnet", "D:\\Data\\Knowledge Embedding\\Wordnet\\", "train.txt", "dev.txt", "test.txt", false);
const Dataset Freebase("Freebase", "D:\\Data\\Knowledge Embedding\\Freebase\\", "train.txt", "dev.txt", "test.txt", false);
const string report_path = "D:\\สตั้\\Report\\Experiment.Embedding\\";
