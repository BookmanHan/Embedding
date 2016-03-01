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

using namespace std;
using namespace arma;
using namespace boost;

template<typename T>
class vmat_storage
{
public:
	static void save(const vector<Mat<T>>& vmatout, ofstream& fout)
	{
		auto n_size = vmatout.size();
		fout.write((char*)&n_size, sizeof(vmatout.size()));

		for (const Mat<T> & ivmatout : vmatout)
		{
			fout.write((char*)&ivmatout.n_rows, sizeof(ivmatout.n_rows));
			fout.write((char*)&ivmatout.n_cols, sizeof(ivmatout.n_cols));

			fout.write((char*)ivmatout.memptr(), ivmatout.n_elem * sizeof(T));
		}
	}

	static void load(vector<Mat<T>>& vmatin, ifstream& fin)
	{
		vector<Mat<T>>::size_type n_size;
		fin.read((char*)&n_size, sizeof(n_size));
		vmatin.resize(n_size);

		for (Mat<T> & ivmatin : vmatin)
		{
			arma::uword	n_row, n_col;

			fin.read((char*)&n_row, sizeof(n_row));
			fin.read((char*)&n_col, sizeof(n_col));

			ivmatin.resize(n_row, n_col);
			fin.read((char*)ivmatin.memptr(), n_row * n_col * sizeof(T));
		}
	}

public:
	static void save(const vector<Col<T>>& vmatout, ofstream& fout)
	{
		auto n_size = vmatout.size();
		fout.write((char*)&n_size, sizeof(vmatout.size()));

		for (const Col<T> & ivmatout : vmatout)
		{
			fout.write((char*)&ivmatout.n_rows, sizeof(ivmatout.n_rows));
			fout.write((char*)&ivmatout.n_cols, sizeof(ivmatout.n_cols));

			fout.write((char*)ivmatout.memptr(), ivmatout.n_elem * sizeof(T));
		}
	}

	static void load(vector<Col<T>>& vmatin, ifstream& fin)
	{
		vector<Col<T>>::size_type n_size;
		fin.read((char*)&n_size, sizeof(n_size));
		vmatin.resize(n_size);

		for (Col<T> & ivmatin : vmatin)
		{
			arma::uword	n_row, n_col;

			fin.read((char*)&n_row, sizeof(n_row));
			fin.read((char*)&n_col, sizeof(n_col));

			ivmatin.resize(n_row);
			fin.read((char*)ivmatin.memptr(), n_row * n_col * sizeof(T));
		}
	}
};

class web_page_stream
	:public ostream
{
protected:
	const string file_name;
	stringstream fout;

public:
	web_page_stream(const string& file_name)
		:file_name(file_name)
	{
		;
	}

public:
	template<typename T>
	web_page_stream& operator << (T thing)
	{
		fout << thing;
		return *this;
	}

	template<typename T>
	web_page_stream& operator << (decltype(endl<T>) thing)
	{
		newline();
		return *this;
	}

public:
	void newline()
	{
		fout << "<br/>";
	}

	void out()
	{
		ofstream ofout(file_name, ios::out);
		ofout << "<html>" << endl;
		ofout << "<head><meta http-equiv=\"refresh\" content=\"1\"></head>" << endl;
		ofout << "<body>" << endl;
		ofout << fout.str() << endl;
		ofout << "</body>" << endl;
		ofout << "</html>";
		ofout.close();
	}

	void show()
	{
		out();
		system(file_name.c_str());
	}

	void cls()
	{
		fout.str("");
	}

	void stress()
	{

	}

	~web_page_stream()
	{
		ofstream ofout(file_name, ios::out);
		ofout << "<html>" << endl;
		ofout << "<body>" << endl;
		ofout << fout.str() << endl;
		ofout << "</body>" << endl;
		ofout << "</html>";
		ofout.close();
	}
};

web_page_stream wout("D:\\Temp\\1.html");

void freebase_LSI()
{
	const int dim = 100;
	const double alpha = 0.02;

	vector<pair<string, vector<string>>>	documents;
	fstream fin("D:\\Data\\Knowledge Embedding\\FB15KZS\\description.txt");
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
	fin.close();

	for (auto i = 0; i < 20; ++i)
	{
		wout.cls();
		wout << "Epos : " << i;
		wout.newline();

#pragma omp parallel for
		for (auto idoc = documents.begin(); idoc != documents.end(); ++idoc)
		{
			vec& v_doc = topic_documents[idoc - documents.begin()];
			vec v_doc_grad = zeros(dim);

			for (auto iword = idoc->second.begin(); iword < idoc->second.end(); ++iword)
			{
				vec& v_word = topic_words[*iword];
				v_doc_grad += alpha * as_scalar(1 - v_doc.t()*v_word) * v_word;
				v_word += alpha * as_scalar(1 - v_doc.t()*v_word) * v_doc;

				vec& v_word_ns = topic_words[words[rand() % words.size()]];
				v_doc_grad += alpha * as_scalar(0 - v_doc.t()*v_word_ns) * v_word_ns;
				v_word_ns += alpha * as_scalar(0 - v_doc.t()*v_word_ns) * v_doc;

				v_word_ns = normalise(v_word_ns);
				v_word = normalise(v_word);
			}

			v_doc += v_doc_grad;
			v_doc = normalise(v_doc);
		}

		wout << topic_documents[2].t();
		wout << topic_documents[10].t();

		wout.out();
	}

	for (auto idoc = documents.begin(); idoc != documents.end(); ++idoc)
	{
		vec& v_doc = topic_documents[idoc - documents.begin()];
		v_doc = normalise(v_doc);
	}

	ofstream fout("D:\\Data\\Knowledge Embedding\\FB15KZS\\topics.bsd", ios::binary);
	vmat_storage<double>::save(topic_documents, fout);
	fout.close();
}

void wordnet_LSI()
{
	const int dim = 100;
	const double alpha = 0.02;

	vector<pair<string, vector<string>>>	documents;
	fstream fin("D:\\Data\\Knowledge Embedding\\WN18\\definitions.txt");
	char_separator<char> sep(" \t \"\',.\\?!#%@;:<>()&*");

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
				++i;
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
	fin.close();

	for (auto i = 0; i < 100; ++i)
	{
		cout << "Epos : " << i << endl;

		for (auto idoc = documents.begin(); idoc != documents.end(); ++idoc)
		{
			vec& v_doc = topic_documents[idoc - documents.begin()];
			vec v_doc_grad = zeros(dim);

			for (auto iword = idoc->second.begin(); iword < idoc->second.end(); ++iword)
			{
				vec& v_word = topic_words[*iword];
				v_doc_grad += alpha * as_scalar(1 - v_doc.t()*v_word) * v_word;
				v_word += alpha * as_scalar(1 - v_doc.t()*v_word) * v_doc;

				vec& v_word_ns = topic_words[words[rand() % words.size()]];
				v_doc_grad += alpha * as_scalar(0 - v_doc.t()*v_word_ns) * v_word_ns;
				v_word_ns += alpha * as_scalar(0 - v_doc.t()*v_word_ns) * v_doc;

				v_word_ns = normalise(v_word_ns);
				v_word = normalise(v_word);
			}

			v_doc += v_doc_grad;
			v_doc = normalise(v_doc);
		}

		cout << topic_documents[2].t();
		cout << topic_documents[10].t();
	}

	for (auto idoc = documents.begin(); idoc != documents.end(); ++idoc)
	{
		vec& v_doc = topic_documents[idoc - documents.begin()];
		v_doc = normalise(v_doc);
	}

	fstream fout("D:\\Data\\Knowledge Embedding\\WN18\\topics.bsd", ios::out);
	for (auto i = documents.begin(); i != documents.end(); ++i)
	{
		fout << i->first << endl;
		topic_documents[i - documents.begin()].save(fout, raw_ascii);
	}
} 

int main()
{
	wout.show();
	freebase_LSI();

	return 0;
}

