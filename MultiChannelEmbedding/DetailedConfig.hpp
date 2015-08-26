#pragma once
#include "Import.hpp"
#include "ModelConfig.hpp"

static Dataset FB15K("G:\\Data\\Knowledge Embedding\\FB15K\\", "train.txt", "dev.txt", "test.txt", true);
static Dataset FB13("G:\\Data\\Knowledge Embedding\\FB13\\", "train.txt", "dev.txt", "test.txt", false);
static Dataset WN11("G:\\Data\\Knowledge Embedding\\WN11\\", "train.txt", "dev.txt", "test.txt", false);
static Dataset WN18("G:\\Data\\Knowledge Embedding\\WN18\\", "train.txt", "dev.txt", "test.txt", true);