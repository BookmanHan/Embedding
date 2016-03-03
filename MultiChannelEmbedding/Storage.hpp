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