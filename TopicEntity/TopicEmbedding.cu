#define VIENNACL_WITH_CUDA 1
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
#include <boost/format.hpp>  
#include <boost/tokenizer.hpp>  
#include <boost/algorithm/string.hpp>  
#include <viennacl/matrix.hpp>
#include <viennacl/scalar.hpp>
#include <viennacl/vector.hpp>

using namespace std;
using namespace arma;
using namespace boost;

namespace vn = viennacl;

int main()
{
	vn::matrix<float> mat_A;
	vn::vector<float> vec_b;
	vn::vector<float> vec_x;

	mat A = randu(1000, 1000);
	vec b = randu(1000);

	while (true)
	{
		copy(b.begin(), b.end(), b.begin());

		vec_x = vn::linalg::prod(mat_A, vec_b);

		cout << 'a';
	}

	return 0;
}

