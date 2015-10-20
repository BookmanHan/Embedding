#pragma once
#include "Import.hpp"
#include "ModelConfig.hpp"

const Dataset FB15K("FB15K", "G:\\Data\\Knowledge Embedding\\FB15K\\", "train.txt", "dev.txt", "test.txt", true);
const Dataset FB13("FB13", "G:\\Data\\Knowledge Embedding\\FB13\\", "train.txt", "dev.txt", "test.txt", false);
const Dataset WN11("WN11", "G:\\Data\\Knowledge Embedding\\WN11\\", "train.txt", "dev.txt", "test.txt", false);
const Dataset WN18("WN18", "G:\\Data\\Knowledge Embedding\\WN18\\", "train.txt", "dev.txt", "test.txt", true);
const Dataset Wordnet("Wordnet", "G:\\Data\\Knowledge Embedding\\Wordnet\\", "train.txt", "dev.txt", "test.txt", false);
const Dataset Freebase("Freebase", "G:\\Data\\Knowledge Embedding\\Freebase\\", "train.txt", "dev.txt", "test.txt", false);
const string report_path = "G:\\สตั้\\Report\\Experiment.Embedding\\";
