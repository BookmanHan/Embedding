#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <armadillo>
#include <map>
#include <set>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <iomanip>
#include <bitset>
#include <queue>
#include <boost/function.hpp>
#include <iterator>
#include "Storage.hpp"

using namespace std;
using namespace arma;

inline
string& replace_all(string&   str,const   string&   old_value,const   string&   new_value)   
{   
	while(true)   {   
		string::size_type   pos(0);   
		if(   (pos=str.find(old_value))!=string::npos   )   
			str.replace(pos,old_value.length(),new_value);   
		else   break;   
	}   
	return   str;   
}   

inline 
double sign(const double& x)
{
	if (x==0)
		return 0;
	else
		return x>0?+1:-1;
}

inline
double norm_L2(const vec& m)
{
	return norm(m, 2);
}

void message(const string& strout)
{
	system((string("mshta vbscript:msgbox(\"")
		+ strout
		+ string("\")(window.close)")).c_str());
}