#pragma once
#include "Import.hpp"

using namespace std;
using namespace arma;

template<typename T>
class storage_vmat
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
		arma::uword n_size;
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
		arma::uword n_size;
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

template<typename T>
class storage_vec
{
public:
	static void save(const Col<T>& vscr, ofstream& fout)
	{
		fout.write((char*)&vscr.n_rows, sizeof(vscr.n_rows));
		fout.write((char*)&vscr.n_cols, sizeof(vscr.n_cols));

		fout.write((char*)vscr.memptr(), vscr.n_elem * sizeof(T));
	}

	static void load(Col<T>& vscr, ifstream& fout)
	{
		arma::uword	n_row, n_col;

		fout.read((char*)&n_row, sizeof(n_row));
		fout.read((char*)&n_col, sizeof(n_col));

		vscr.resize(n_row);
		fout.read((char*)vscr.memptr(), n_row * n_col * sizeof(T));
	}
};

template<typename T>
class storage_vector
{
public:
	static void save(const vector<T>& vscr, ofstream& fout)
	{
		auto n_size = vmatout.size();
		fout.write((char*)&n_size, sizeof(vmatout.size()));

		for (auto i = vscr.begin(); i != vscr.end(); ++i)
		{
			fout.write((char*)&(*i), sizeof(T));
		}
	}

	static void load(vector<T>& vscr, ifstream& fout)
	{
		arma::uword	n_size;
		fout.read((char*)&n_size, sizeof(vmatout.size()));

		vscr.resize(n_size);
		for (auto i = vscr.begin(); i != vscr.end(); ++i)
		{
			fout.write((char*)&(*i), sizeof(T));
		}
	}
};

class storage_vstring
{
public:
	static void save(const vector<string>& vscr, ofstream& fout)
	{
		auto n_size = vscr.size();
		fout.write((char*)&n_size, sizeof(vscr.size()));

		for (auto i = vscr.begin(); i != vscr.end(); ++i)
		{
			auto n_len = i->length();
			fout.write((char*)&n_len, sizeof(n_len));
			fout.write((char*)i->data(), sizeof(char)*n_len);
		}
	}

	static void load(vector<string>& vscr, ifstream& fout)
	{
		auto n_size = vscr.size();
		fout.read((char*)&n_size, sizeof(vscr.size()));

		vscr.resize(n_size);
		for (auto i = vscr.begin(); i != vscr.end(); ++i)
		{
			auto n_len = i->length();
			fout.read((char*)&n_len, sizeof(n_len));

			char* ca_str = new char(n_len+1);
			fout.read((char*)ca_str, sizeof(char)*n_len);
			ca_str[n_len] = 0;
			*i = ca_str;

			delete[] ca_str;
		}
	}
};