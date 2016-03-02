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
#include <viennacl/vector.hpp>
#include <viennacl/matrix.hpp>

using namespace std;
using namespace arma;
using namespace boost;

typedef viennacl::scalar<float>	ScalarType;

int main()
{
	std::vector<ScalarType>      stl_vec(1000);
	viennacl::vector<ScalarType> vcl_vec(100000);

	for (size_t i = 0; i < stl_vec.size(); ++i)
		stl_vec[i] = i;

	int a;
	copy(stl_vec.begin(), stl_vec.end(), vcl_vec.begin());

	cout << "Ready";
	cin >> a;
	copy(vcl_vec.begin(), vcl_vec.end(), stl_vec.begin());

	return 0;
}

