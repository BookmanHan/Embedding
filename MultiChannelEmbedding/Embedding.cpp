#include "Model.hpp"
#include "MultiChannel_Innerproduct.hpp"

int main(int argc, char* argv[])
{
	EmbeddingModel*	Model = new MultiChannel_Innerproduct(10, 0.01);
	Model->run();

	return 0;
}