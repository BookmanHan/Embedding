#include <vector>
#include <map>
#include <set>
#include <cstdlib>
#include <cmath>
#include <armadillo>
#include <iostream>
#include <fstream>
#include <boost/format.hpp>  
#include <boost/tokenizer.hpp>  
#include <boost/algorithm/string.hpp>  

using namespace std;
using namespace arma;
using namespace boost;

int main()
{
	const int dim = 100;
	const double alpha = 0.02;

	vector<pair<string, vector<string>>>	documents;
	fstream fin("D:\\Data\\Knowledge Embedding\\FB15K\\description.txt");
	char_separator<char> sep(" \t \"\',.\\?!#%@");

	vector<vec>			topic_documents;
	vector<string>		words;
	map<string, vec>	topic_words;

	while (!fin.eof())
	{
		string strin;
		getline(fin, strin);
		tokenizer<char_separator<char>>	token(strin, sep);

		string entity_name;
		vector<string>	entity_description;
		for (auto i = token.begin(); i != token.end(); ++i)
		{
			if (i == token.begin())
			{
				entity_name = *i;
			}
			else
			{
				entity_description.push_back(*i);
				if (topic_words.find(*i) == topic_words.end())
				{
					topic_words[*i] = randu(dim);
					words.push_back(*i);
				}
			}
		}

		documents.push_back(make_pair(entity_name, entity_description));
		topic_documents.push_back(randu(dim));
	}

	for (auto i = 0; i < 20; ++i)
	{ 
		cout << "Epos : " << i << endl;

		for (auto idoc = documents.begin(); idoc != documents.end(); ++idoc)
		{
			vec& v_doc = topic_documents[idoc - documents.begin()];
			vec v_doc_grad = zeros(dim);

//#pragma omp parallel for
			for (auto iword = idoc->second.begin(); iword != idoc->second.end(); ++iword)
			{
				vec& v_word = topic_words[*iword];
				v_doc_grad += alpha * as_scalar(1 - v_doc.t()*v_word) * v_word;
				v_word += alpha * as_scalar(1 - v_doc.t()*v_word) * v_doc;

				vec& v_word_ns = topic_words[words[rand() % words.size()]];
				v_doc_grad -= alpha * as_scalar(1 - v_doc.t()*v_word_ns) * v_word_ns;
				v_word_ns -= alpha * as_scalar(1 - v_doc.t()*v_word_ns) * v_doc;

				v_word_ns = max(v_word_ns, zeros(dim));
				v_word = max(v_word, zeros(dim));

				v_word_ns = normalise(v_word_ns);
				v_word = normalise(v_word);
			}

			v_doc += v_doc_grad;
			v_doc = max(v_doc, zeros(dim));
			v_doc = normalise(v_doc);
		}

		cout << topic_documents[2].t();
	}

	for (auto idoc = documents.begin(); idoc != documents.end(); ++idoc)
	{
		vec& v_doc = topic_documents[idoc - documents.begin()];
		v_doc = normalise(v_doc);
	}

	fstream fout("D:\\Data\\Knowledge Embedding\\FB15K\\topics.100.bsd", ios::out);
	for (auto i = documents.begin(); i != documents.end(); ++i)
	{
		fout << i->first << endl;
		topic_documents[i - documents.begin()].save(fout, raw_ascii);
	}

	return 0;
}

