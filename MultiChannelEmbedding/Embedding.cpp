#define Wordnet
#define LP
#include "Model.hpp"
#include "Embedding_MultiChannel.hpp"
#include "Embedding_Geometric.hpp"
#include <cstdlib>
#include <omp.h>

//TransG(50, 5, 8, 0.001) @ WN18.Round5000 = 468, 457, 82.5, 94.4

/*
1.Triplets Classification.
TransG(50, 5, 2000, 0.01, 0) @ WN11.Round.1000 = 87.4%
	_type_of:0.887384
	_synset_domain_topic:0.956945
	_has_instance:0.808232
	_member_holonym:0.92347
	_part_of:0.899415
	_has_part:0.887596
	_member_meronym:0.920053
	_similar_to:0.605238 (0.715)
	_subordinate_instance_of:0.93
	_domain_region:0.776892
	_domain_topic:0.794483

TransG(200, 4, exp(2), 0.00005, 0) @ FB13.Round.1000 = 84.4 %
	religion:0.843033
	cause_of_death:0.799562
	profession:0.851754
	gender:0.84705
	nationality:0.892274
	institution:0.746454
	ethnicity:0.797003

TransGA(50, 5, 500, 2000,  0.00001, 0.01) @ WN11.Round500 = 87.7 %

2. Link Prediction
TransG(50, 5, exp(2), 0.001, 0) @ WN18.Round.5000 = 478, 466, 82.6, 94.3

*/
int main(int argc, char* argv[])
{
	omp_set_num_threads(8);
	EmbeddingModel*	model = new TransG(50, 5, exp(2), 0.001, 0);
	model->run(10000);

	return 0;
}