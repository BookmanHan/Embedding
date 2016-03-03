#define VIENNACL_WITH_CUDA
#include <vector>
#include <map>
#include <set>
#include <cstdlib>
#include <cmath>
#include <armadillo>
#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <sstream>
#include <viennacl/matrix.hpp>
#include <viennacl/scalar.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/fft.hpp>
#include <viennacl/linalg/fft_operations.hpp>

using namespace std;
using namespace arma;

namespace vcl = viennacl;

int main()
{
	vcl::matrix<float> mat_a(1000, 1000);
	vcl::matrix<float> mat_b(1000, 1000);

	while (true)
	{
		vcl::fft(mat_a, mat_b);
		cout << 'a';
	}

	return 0;
}

