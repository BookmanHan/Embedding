#pragma once
#include "Model.hpp"

#pragma once
#include "Model.hpp"

inline double sign(const double& x)
{
	return x>0?+1:-1;
}

class GeometricEmbeddingHadamard
	:public GeometricEmbeddingModel
{
public:
	GeometricEmbeddingHadamard(int dim, double alpha)
		:GeometricEmbeddingModel(dim, alpha)
	{
		;
	}

public:
	virtual double prob_triplets( const pair<pair<string, string>,string>& triplet )
	{
		vec error = embedding_entity[name_entity[triplet.first.first]]
			% embedding_entity[name_entity[triplet.first.second]]
			% embedding_relation[name_relation[triplet.second]];

		return sum(error);
	}

	virtual double train_once( const pair<pair<string, string>,string>& triplet, double factor )
	{
		vec& head = embedding_entity[name_entity[triplet.first.first]];
		vec& tail = embedding_entity[name_entity[triplet.first.second]];
		vec& relation = embedding_relation[name_relation[triplet.second]];

		pair<pair<string, string>,string> triplet_f;
		sample_false_triplet(triplet, triplet_f);
		vec& head_f = embedding_entity[name_entity[triplet_f.first.first]];
		vec& tail_f = embedding_entity[name_entity[triplet_f.first.second]];
		vec& relation_f = embedding_relation[name_relation[triplet_f.second]];

		head += alpha * tail % relation;
		tail += alpha * head % relation;
		relation += alpha * head % tail;
		head_f -= alpha * tail_f % relation_f;
		tail_f -= alpha * head_f % relation_f;
		relation_f -= alpha * head_f % tail_f;

		head = normalise(head);
		tail = normalise(tail);
		relation = normalise(relation);
		head_f = normalise(head_f);
		tail_f = normalise(tail_f);
		relation_f = normalise(relation_f);
	}
};

class TransE
	:public GeometricEmbeddingModel
{
public:
	TransE(int dim, double alpha)
		:GeometricEmbeddingModel(dim, alpha)
	{
		;
	}

public:
	virtual double prob_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		vec error = embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- embedding_entity[triplet.first.second];

		return - sum(abs(error));
	}

	virtual double train_once( const pair<pair<unsigned, unsigned>,unsigned>& triplet, double factor )
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];

		pair<pair<unsigned, unsigned>,unsigned> triplet_f;
		sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > 1)
			return 0;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];

		head -= alpha * sign(head + relation - tail);
		tail += alpha * sign(head + relation - tail);
		relation -= alpha * sign(head + relation - tail);
		head_f += alpha * sign(head_f + relation_f - tail_f);
		tail_f -= alpha * sign(head_f + relation_f - tail_f);
		relation_f += alpha * sign(head_f + relation_f - tail_f);

		head = normalise(head);
		tail = normalise(tail);
		relation = normalise(relation);
		head_f = normalise(head_f);
		tail_f = normalise(tail_f);
		relation_f = normalise(relation_f);
	}
};

class TransA
	:public TransE
{
protected:
	vector<mat>	mat_r;

public:
	TransA(int dim, double alpha)
		:TransE(dim, alpha)
	{
		mat_r.resize(set_relation.size());
		for_each(mat_r.begin(), mat_r.end(), [&](mat& m){ m = eye(dim,dim);});
	}

public:
	virtual double prob_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		vec error = embedding_entity[triplet.first.first]
		+ embedding_relation[triplet.second]
		- embedding_entity[triplet.first.second];

		return - as_scalar(abs(error).t()*mat_r[triplet.second]*abs(error));
	}

	virtual double train_once( const pair<pair<unsigned, unsigned>,unsigned>& triplet, double factor )
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];

		pair<pair<unsigned, unsigned>,unsigned> triplet_f;
		sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > 1)
			return 0;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];

		head -= alpha * sign(head + relation - tail);
		tail += alpha * sign(head + relation - tail);
		relation -= alpha * sign(head + relation - tail);
		head_f += alpha * sign(head_f + relation_f - tail_f);
		tail_f -= alpha * sign(head_f + relation_f - tail_f);
		relation_f += alpha * sign(head_f + relation_f - tail_f);

		head = normalise(head);
		tail = normalise(tail);
		relation = normalise(relation);
		head_f = normalise(head_f);
		tail_f = normalise(tail_f);
		relation_f = normalise(relation_f);

		mat_r[triplet.second] -= abs(head + relation - tail) * abs(head + relation - tail).t()
			- abs(head_f + relation_f - tail_f) * abs(head_f + relation_f - tail_f).t();
	}

	virtual void train( double alpha )
	{
		for_each(mat_r.begin(), mat_r.end(), [&](mat& m){ m = eye(dim,dim);});
		TransE::train(alpha);
	}
};

class TransA2
	:public TransE
{
protected:
	vector<mat>	mat_r;

public:
	TransA2(int dim, double alpha)
		:TransE(dim, alpha)
	{
		mat_r.resize(set_relation.size());
		for_each(mat_r.begin(), mat_r.end(), [&](mat& m){ m = eye(dim,dim);});
	}

public:
	virtual double prob_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		vec error = embedding_entity[triplet.first.first]
		+ embedding_relation[triplet.second]
		- embedding_entity[triplet.first.second];

		return - as_scalar(abs(error).t()*mat_r[triplet.second]*abs(error));
	}

	virtual double probability_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		vec error = embedding_entity[triplet.first.first]
		+ embedding_relation[triplet.second]
		- embedding_entity[triplet.first.second];

		return - sum(abs(error));
	}

	virtual double train_once( const pair<pair<unsigned, unsigned>,unsigned>& triplet, double factor )
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];

		pair<pair<unsigned, unsigned>,unsigned> triplet_f;
		sample_false_triplet(triplet, triplet_f);

		if (probability_triplets(triplet) - probability_triplets(triplet_f) > 2)
			return 0;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];

		head -= alpha * sign(head + relation - tail);
		tail += alpha * sign(head + relation - tail);
		relation -= alpha * sign(head + relation - tail);
		head_f += alpha * sign(head_f + relation_f - tail_f);
		tail_f -= alpha * sign(head_f + relation_f - tail_f);
		relation_f += alpha * sign(head_f + relation_f - tail_f);

		head = normalise(head);
		tail = normalise(tail);
		relation = normalise(relation);
		head_f = normalise(head_f);
		tail_f = normalise(tail_f);
		relation_f = normalise(relation_f);
	}

	virtual void train( double alpha )
	{
		TransE::train(alpha);
		
		if (best_result >= 0.8)
		{
			for_each(mat_r.begin(), mat_r.end(), [&](mat& m){ m = eye(dim,dim);});
			for(auto i=i_data_train.begin(); i!=i_data_train.end(); ++i)
			{
				auto triplet = *i;

				vec& head = embedding_entity[triplet.first.first];
				vec& tail = embedding_entity[triplet.first.second];
				vec& relation = embedding_relation[triplet.second];

				pair<pair<unsigned, unsigned>,unsigned> triplet_f;
				sample_false_triplet(triplet, triplet_f);

				vec& head_f = embedding_entity[triplet_f.first.first];
				vec& tail_f = embedding_entity[triplet_f.first.second];
				vec& relation_f = embedding_relation[triplet_f.second];		

				mat_r[triplet.second] -= abs(head + relation - tail) * abs(head + relation - tail).t()
					- abs(head_f + relation_f - tail_f) * abs(head_f + relation_f - tail_f).t();
			}
			for_each(mat_r.begin(), mat_r.end(), [=](mat& elem){elem = normalise(elem);});
		}
	}
};

class TransG
	:public GeneralGeometricEmbeddingModel
{
public:
	TransG(int dim, double alpha)
		:GeneralGeometricEmbeddingModel(dim, alpha)
	{
		;
	}

public:
	virtual double prob_triplets( const pair<pair<string, string>,string>& triplet )
	{
		vec error = embedding_entity[name_entity[triplet.first.first]] 
		+ embedding_relation[name_relation[triplet.second]] 
		- embedding_entity[name_entity[triplet.first.second]];

		return - as_scalar(abs(error).t() * mat_relation[name_relation[triplet.second]] * abs(error));
	}

	virtual double train_once( const pair<pair<string, string>,string>& triplet, double factor )
	{
		vec& head = embedding_entity[name_entity[triplet.first.first]];
		vec& tail = embedding_entity[name_entity[triplet.first.second]];
		vec& relation = embedding_relation[name_relation[triplet.second]];

		pair<pair<string, string>,string> triplet_f;
		sample_false_triplet(triplet, triplet_f);
		vec& head_f = embedding_entity[name_entity[triplet_f.first.first]];
		vec& tail_f = embedding_entity[name_entity[triplet_f.first.second]];
		vec& relation_f = embedding_relation[name_relation[triplet_f.second]];

		head -= alpha * sign(head + relation - tail);
		tail += alpha * sign(head + relation - tail);
		relation -= alpha * sign(head + relation - tail);
		head_f += alpha * sign(head_f + relation_f - tail_f);
		tail_f -= alpha * sign(head_f + relation_f - tail_f);
		relation_f += alpha * sign(head_f + relation_f - tail_f);

		head = normalise(head);
		tail = normalise(tail);
		relation = normalise(relation);
		head_f = normalise(head_f);
		tail_f = normalise(tail_f);
		relation_f = normalise(relation_f);

		mat_relation[name_relation[triplet.second]] -= abs(head + relation - tail) * abs(head + relation - tail).t()
			- abs(head_f + relation_f - tail_f) * abs(head_f + relation_f - tail_f).t();
	}

	virtual void train(double alpha)
	{
		for_each(mat_relation.begin(), mat_relation.end(), [=](mat& elem){elem = eye(dim, dim);});
		EmbeddingModel::train(alpha);
		for_each(mat_relation.begin(), mat_relation.end(), [=](mat& elem){elem = normalise(elem);});
	}

};

class TransGMP
	:public GeometricEmbeddingModel
{
protected:
	unsigned int sampling_times;
	vec			 error;

public:
	TransGMP(int dim, double alpha, int sampling_times =1)
		:GeometricEmbeddingModel(dim, alpha), 
		sampling_times(sampling_times),
		error(dim, 1)
	{
		;
	}

public:
	virtual double prob_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet ) = 0;
	virtual double probability_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet)
	{
		return prob_triplets(triplet);
	}

public:
	virtual vec grad(const pair<pair<unsigned, unsigned>,unsigned>& triplet, componet part) = 0;

public:
	virtual double pre_train_once( const pair<pair<unsigned, unsigned>,unsigned>& triplet, double factor )
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];

		pair<pair<unsigned, unsigned>,unsigned> triplet_f;
		sample_false_triplet(triplet, triplet_f);

		if (probability_triplets(triplet) -  probability_triplets(triplet_f) > 2)
			return 0;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];

		head -= alpha * sign(head + relation - tail);
		tail += alpha * sign(head + relation - tail);
		relation -= alpha * sign(head + relation - tail);
		head_f += alpha * sign(head_f + relation_f - tail_f);
		tail_f -= alpha * sign(head_f + relation_f - tail_f);
		relation_f += alpha * sign(head_f + relation_f - tail_f);

		head = normalise(head);
		tail = normalise(tail);
		relation = normalise(relation);
		head_f = normalise(head_f);
		tail_f = normalise(tail_f);
		relation_f = normalise(relation_f);
	}

	virtual double train_once( const pair<pair<unsigned, unsigned>,unsigned>& triplet, double factor )
	{
		//if (epos <= 500)
		//	return pre_train_once(triplet, factor);

		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		
		vec head_grad(dim, 1, fill::zeros);
		vec tail_grad(dim, 1, fill::zeros);
		vec relation_grad(dim, 1, fill::zeros);

		double total_normalizor = 0;
		for(auto cnt=0; cnt<sampling_times; ++cnt)
		{
			for(auto i=7; i>=0; --i)
			{
				bool head = i & 0x001;
				bool tail = i & 0x100;
				bool relation = i & 0x010;
				 
				pair<pair<unsigned, unsigned>,unsigned> triplet_sample;
				sample_triplet(triplet, triplet_sample, head, relation, tail);

				double prob = exp(probability_triplets(triplet_sample));
				if (_isnan(prob))
					continue;

				total_normalizor += prob;

				if (head == false)
				{
					head_grad -= prob * grad(triplet_sample, componet::componet_head);
				}
				if (tail == false)
				{
					tail_grad -= prob * grad(triplet_sample, componet::componet_tail);
				}
				if (relation == false)
				{
					relation_grad -= prob * grad(triplet_sample, componet::componet_relation);
				}
			}
		}
		
		if (total_normalizor != 0)
		{
			head_grad /= total_normalizor;
			tail_grad /= total_normalizor;
			relation_grad /= total_normalizor;
		}

		head_grad += grad(triplet, componet::componet_head);
		tail_grad += grad(triplet, componet::componet_tail);
		relation_grad += grad(triplet, componet::componet_relation);

		head -= factor * head_grad;
		tail -= factor * tail_grad;
		relation -= factor * relation_grad;

		head = normalise(head);
		tail = normalise(tail);
		relation = normalise(relation);

		pair<pair<unsigned, unsigned>,unsigned> triplet_f;
		sample_false_triplet(triplet, triplet_f);

		if (probability_triplets(triplet) -  probability_triplets(triplet_f) > 2)
			return 0;

		if (factor > 0)
			train_once(triplet_f, -factor);
	}
};

class TransGGMP
	:public GeneralGeometricEmbeddingModel
{
protected:
	unsigned int sampling_times;
	vec			 error;

public:
	TransGGMP(int dim, double alpha, int sampling_times =1)
		:GeneralGeometricEmbeddingModel(dim, alpha), 
		sampling_times(sampling_times),
		error(dim, 1)
	{
		;
	}

public:
	virtual double prob_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet ) = 0;
	virtual double probability_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet, vec & error)
	{
		return prob_triplets(triplet);
	}

public:
	virtual vec grad(const pair<pair<unsigned, unsigned>,unsigned>& triplet, componet part) = 0;
	virtual mat grad_matr(const pair<pair<unsigned, unsigned>,unsigned>& triplet, componet part) = 0;

public:
	virtual double train_once( const pair<pair<unsigned, unsigned>,unsigned>& triplet, double factor )
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		vec head_grad(dim, 1, fill::zeros);
		vec tail_grad(dim, 1, fill::zeros);
		vec relation_grad(dim, 1, fill::zeros);
		mat mat_grad(dim, dim, fill::zeros);

		double total_normalizor = 0;
		for(auto cnt=0; cnt<sampling_times; ++cnt)
		{
			for(unsigned i=0; i<8; ++i)
			{
				bool head = i & 0x001;
				bool tail = i & 0x100;
				bool relation = i & 0x010;

				pair<pair<unsigned, unsigned>,unsigned> triplet_sample;
				sample_triplet(triplet, triplet_sample, head, relation, tail);

				double prob = exp(probability_triplets(triplet_sample, error));
				if (_isnan(prob))
					continue;
				total_normalizor += prob;

				if (head == false)
				{
					head_grad -= prob * grad(triplet_sample, componet::componet_head);
				}
				if (tail == false)
				{
					tail_grad -= prob * grad(triplet_sample, componet::componet_tail);
				}
				if (relation == false)
				{
					relation_grad -= prob * grad(triplet_sample, componet::componet_relation);
				}
				
				mat_grad -= prob * grad_matr(triplet_sample, GeometricEmbeddingModel::componet_matr);
			}
		}

		if (total_normalizor != 0)
		{
			head_grad /= total_normalizor;
			tail_grad /= total_normalizor;
			relation_grad /= total_normalizor;
			mat_grad /= total_normalizor;
		}

		head_grad += grad(triplet, GeometricEmbeddingModel::componet_head);
		tail_grad += grad(triplet, GeometricEmbeddingModel::componet_tail);
		relation_grad += grad(triplet, GeometricEmbeddingModel::componet_relation);
		mat_grad += grad_matr(triplet, GeometricEmbeddingModel::componet_matr);

		head -= alpha * head_grad;
		tail -= alpha * tail_grad;
		relation -= alpha * relation_grad;
		mat_relation[triplet.second] -= alpha * mat_grad;

		//head = normalise(head);
		//tail = normalise(tail);
		//relation = normalise(relation);
	}
};

class TransGMPE
	:public TransGMP
{
public:
	TransGMPE(int dim, double alpha, int sampling_times = 2)
		:TransGMP(dim, alpha, sampling_times)
	{
		;
	}

public:
	virtual double prob_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		return - sum(abs(
			embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- embedding_entity[triplet.first.second]));
	}

	virtual vec grad( const pair<pair<unsigned, unsigned>,unsigned>& triplet, componet part )
	{
		switch(part)
		{
		case GeometricEmbeddingModel::componet_head:
			return sign(embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- embedding_entity[triplet.first.second]);
			break;
		case GeometricEmbeddingModel::componet_tail:
			return - sign(embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- embedding_entity[triplet.first.second]);
			break;
		case GeometricEmbeddingModel::componet_relation:
			return sign(embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- embedding_entity[triplet.first.second]);
			break;
		}
	}

	virtual double probability_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet)
	{
		return - sum(abs(
			embedding_entity[triplet.first.first]
		+ embedding_relation[triplet.second]
		- embedding_entity[triplet.first.second]));
	}
};

class TransGMPEN2
	:public TransGMP
{
public:
	TransGMPEN2(int dim, double alpha, int sampling_times = 2)
		:TransGMP(dim, alpha, sampling_times)
	{
		;
	}

public:
	virtual double prob_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		double error_norm = norm(
			embedding_entity[triplet.first.first]
		+ embedding_relation[triplet.second]
		- embedding_entity[triplet.first.second]);

		return - error_norm ;
	}

	virtual vec grad( const pair<pair<unsigned, unsigned>,unsigned>& triplet, componet part )
	{
		switch(part)
		{
		case GeometricEmbeddingModel::componet_head:
			return embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- embedding_entity[triplet.first.second];
			break;
		case GeometricEmbeddingModel::componet_tail:
			return - (embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- embedding_entity[triplet.first.second]);
			break;
		case GeometricEmbeddingModel::componet_relation:
			return embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- embedding_entity[triplet.first.second];
			break;
		}
	}
};

class TransGMPCosine
	:public TransGMP
{
public:
	TransGMPCosine(int dim, double alpha, int sampling_times = 2)
		:TransGMP(dim, alpha, sampling_times)
	{
		;
	}

public:
	virtual double prob_triplets( const pair<pair<string, string>,string>& triplet )
	{
		return - as_scalar(
			embedding_entity[name_entity[triplet.first.first]].t()
			* embedding_relation[name_relation[triplet.second]]
			- embedding_entity[name_entity[triplet.first.first]].t()
			* embedding_entity[name_relation[triplet.second]]
			- embedding_relation[name_relation[triplet.second]].t()
			* embedding_entity[name_entity[triplet.first.second]]);
	}

	virtual vec grad( const pair<pair<string, string>,string>& triplet, componet part )
	{
		switch(part)
		{
		case GeometricEmbeddingModel::componet_head:
			return (embedding_relation[name_relation[triplet.second]]
			- embedding_entity[name_entity[triplet.first.second]]);
			break;
		case GeometricEmbeddingModel::componet_tail:
			return (- embedding_entity[name_entity[triplet.first.first]]
			- embedding_relation[name_relation[triplet.second]]);
			break;
		case GeometricEmbeddingModel::componet_relation:
			return (embedding_entity[name_entity[triplet.first.first]]
			- embedding_entity[name_entity[triplet.first.second]]);
			break;
		}
	}
};

class TransGMPA
	:public TransGMP
{
protected:
	vector<mat>	mat_r;

public:
	TransGMPA(int dim, double alpha, int sampling_times=1)
		:TransGMP(dim, alpha, sampling_times)
	{
		mat_r.resize(set_relation.size());
		for_each(mat_r.begin(), mat_r.end(), [=](mat& m){m=eye(dim,dim);});
	}

public:
	virtual vec grad( const pair<pair<unsigned, unsigned>,unsigned>& triplet, componet part )
	{
		switch(part)
		{
		case GeometricEmbeddingModel::componet_head:
			return sign(embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- embedding_entity[triplet.first.second]);
			break;
		case GeometricEmbeddingModel::componet_tail:
			return - sign(embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- embedding_entity[triplet.first.second]);
			break;
		case GeometricEmbeddingModel::componet_relation:
			return sign(embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- embedding_entity[triplet.first.second]);
			break;
		}
	}

	virtual double prob_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		vec error = embedding_entity[triplet.first.first]
		+ embedding_relation[triplet.second]
		- embedding_entity[triplet.first.second];

		return  - as_scalar(abs(error).t()*mat_r[triplet.second]*abs(error));
	}

	virtual void train( double alpha )
	{
		GeometricEmbeddingModel::train(alpha);

		if (best_result >= 0.80)
		{
			for_each(mat_r.begin(), mat_r.end(), [&](mat& m){m=eye(dim,dim);});
			for(auto i=i_data_train.begin(); i!=i_data_train.end(); ++i)
			{
				auto& triplet = *i;
				vec& head = embedding_entity[triplet.first.first];
				vec& tail = embedding_entity[triplet.first.second];
				vec& relation = embedding_relation[triplet.second];

				pair<pair<unsigned, unsigned>,unsigned> triplet_f;
				sample_false_triplet(triplet, triplet_f);
				vec& head_f = embedding_entity[triplet_f.first.first];
				vec& tail_f = embedding_entity[triplet_f.first.second];
				vec& relation_f = embedding_relation[triplet_f.second];

				mat_r[triplet.second] += 
					- abs(head + relation - tail) * abs(head + relation - tail).t()
					+ abs(head_f + relation_f - tail_f) * abs(head_f + relation_f - tail_f).t();
			}
			for_each(mat_r.begin(), mat_r.end(), [=](mat& elem){elem = normalise(elem);});
		}
	}

	virtual double probability_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		return - sum(abs(
			embedding_entity[triplet.first.first]
		+ embedding_relation[triplet.second]
		- embedding_entity[triplet.first.second]));
	}
};

class TransGGMPR
	:public TransGGMP
{
public:
	TransGGMPR(int dim, double alpha, int sampling_times)
		:TransGGMP(dim, alpha, sampling_times)
	{
		;
	}

public:
	virtual vec grad( const pair<pair<unsigned, unsigned>,unsigned>& triplet, componet part )
	{
		switch(part)
		{
		case GeometricEmbeddingModel::componet_head:
			return mat_relation[triplet.second].t() *
				sign(mat_relation[triplet.second] * embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- mat_relation[triplet.second] * embedding_entity[triplet.first.second]);
			break;
		case GeometricEmbeddingModel::componet_tail:
			return - mat_relation[triplet.second].t() *
				sign(mat_relation[triplet.second] * embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- mat_relation[triplet.second] * embedding_entity[triplet.first.second]);
			break;
		case GeometricEmbeddingModel::componet_relation:
			return sign(mat_relation[triplet.second] * embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- mat_relation[triplet.second] * embedding_entity[triplet.first.second]);
			break;
		}
	}

	virtual mat grad_matr( const pair<pair<unsigned, unsigned>,unsigned>& triplet, componet part )
	{
		return sign(mat_relation[triplet.second] * embedding_entity[triplet.first.first]
		+ embedding_relation[triplet.second]
		- mat_relation[triplet.second] * embedding_entity[triplet.first.second])
			* (embedding_entity[triplet.first.first] - embedding_entity[triplet.first.second]).t();
	}

	virtual double prob_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		return - sum(abs(mat_relation[triplet.second] * embedding_entity[triplet.first.first]
		+ embedding_relation[triplet.second]
		- mat_relation[triplet.second] * embedding_entity[triplet.first.second]));
	}

	virtual void train( double alpha )
	{
		for_each(mat_relation.begin(), mat_relation.end(), [=](mat& elem){elem = eye(dim,dim);});
		GeometricEmbeddingModel::train(alpha);
		for_each(mat_relation.begin(), mat_relation.end(), [=](mat& elem){elem = normalise(elem);});
	}
};

class TransGGMPA
	:public TransGGMP
{
public:
	TransGGMPA(int dim, double alpha, int sampling_times=1)
		:TransGGMP(dim, alpha, sampling_times)
	{
		;
	}

public:
	virtual vec grad( const pair<pair<unsigned, unsigned>,unsigned>& triplet, componet part )
	{
		switch(part)
		{
		case GeometricEmbeddingModel::componet_head:
			return sign( embedding_entity[triplet.first.first] + embedding_relation[triplet.second]
			-  embedding_entity[triplet.first.second]);
			break;
		case GeometricEmbeddingModel::componet_tail:
			return - sign( embedding_entity[triplet.first.first] + embedding_relation[triplet.second]
			-  embedding_entity[triplet.first.second]);
			break;
		case GeometricEmbeddingModel::componet_relation:
			return sign( embedding_entity[triplet.first.first] + embedding_relation[triplet.second]
			-  embedding_entity[triplet.first.second]);
			break;
		}
	}

	virtual mat grad_matr( const pair<pair<unsigned, unsigned>,unsigned>& triplet, componet part )
	{
		return abs( embedding_entity[triplet.first.first] + embedding_relation[triplet.second]
		-  embedding_entity[triplet.first.second]) * abs( embedding_entity[triplet.first.first] 
		+ embedding_relation[triplet.second] -  embedding_entity[triplet.first.second]).t();
	}

	virtual double prob_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		return - sum(abs(mat_relation[triplet.second] * embedding_entity[triplet.first.first]
		+ embedding_relation[triplet.second]
		- mat_relation[triplet.second] * embedding_entity[triplet.first.second]));
	}

	virtual void train( double alpha )
	{
		for_each(mat_relation.begin(), mat_relation.end(), [=](mat& elem){elem = eye(dim,dim);});
		GeometricEmbeddingModel::train(alpha);
		for_each(mat_relation.begin(), mat_relation.end(), [=](mat& elem){elem = normalise(elem);});
	}
};


class TransGGMPRA
	:public TransGGMP
{
protected:
	vector<mat>	mat_a;

public:
	TransGGMPRA(int dim, double alpha, int sampling_times)
		:TransGGMP(dim, alpha, sampling_times)
	{
		mat_a.resize(set_relation.size());
		for_each(mat_a.begin(), mat_a.end(), [&](mat& r){r = eye(dim, dim);});
	}

public:
	virtual vec grad( const pair<pair<unsigned, unsigned>,unsigned>& triplet, componet part )
	{
		switch(part)
		{
		case GeometricEmbeddingModel::componet_head:
			return sign(mat_relation[triplet.second] * embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- mat_relation[triplet.second] * embedding_entity[triplet.first.second]);
			break;
		case GeometricEmbeddingModel::componet_tail:
			return - sign(mat_relation[triplet.second] * embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- mat_relation[triplet.second] * embedding_entity[triplet.first.second]);
			break;
		case GeometricEmbeddingModel::componet_relation:
			return sign(mat_relation[triplet.second] * embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- mat_relation[triplet.second] * embedding_entity[triplet.first.second]);
			break;
		}
	}

	virtual mat grad_matr( const pair<pair<unsigned, unsigned>,unsigned>& triplet, componet part )
	{
		switch(part)
		{
		case GeometricEmbeddingModel::componet_matr:
			return sign(mat_relation[triplet.second] * embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- mat_relation[triplet.second]* embedding_entity[triplet.first.second]) 
			* (embedding_entity[triplet.first.first] - embedding_entity[triplet.first.second]).t();
			break;
		case GeometricEmbeddingModel::componet_mata:
			return -  abs(mat_relation[triplet.second] * embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- mat_relation[triplet.second]* embedding_entity[triplet.first.second]) 
				* abs(mat_relation[triplet.second] * embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- mat_relation[triplet.second]* embedding_entity[triplet.first.second]) .t();
		}
	}

	virtual double prob_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		vec error = (mat_relation[triplet.second] * embedding_entity[triplet.first.first]
		+ embedding_relation[triplet.second]
		- mat_relation[triplet.second]* embedding_entity[triplet.first.second]) ;

		return - as_scalar(error.t() * mat_a[triplet.second] * error);
	}

	virtual void train( double alpha )
	{
		for_each(mat_relation.begin(), mat_relation.end(), [=](mat& elem){elem = eye(dim,dim);});
		for_each(mat_a.begin(), mat_a.end(), [=](mat& elem){elem = eye(dim,dim);});
		GeometricEmbeddingModel::train(alpha);
	}

	virtual double train_once( const pair<pair<unsigned, unsigned>,unsigned>& triplet, double factor )
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		vec head_grad(dim, 1, fill::zeros);
		vec tail_grad(dim, 1, fill::zeros);
		vec relation_grad(dim, 1, fill::zeros);
		mat mat_grad(dim, dim, fill::zeros);
		mat mata_grad(dim, dim, fill::zeros);

		double total_normalizor = 0;
		for(auto cnt=0; cnt<sampling_times; ++cnt)
		{
			for(unsigned i=0; i<8; ++i)
			{
				bool head = i & 0x001;
				bool tail = i & 0x100;
				bool relation = i & 0x010;

				pair<pair<unsigned, unsigned>,unsigned> triplet_sample;
				sample_triplet(triplet, triplet_sample, head, relation, tail);

				double prob = exp(probability_triplets(triplet_sample, error));
				if (_isnan(prob))
					continue;

				if (head == false)
				{
					head_grad -= prob * grad(triplet_sample, componet::componet_head);
				}
				if (tail == false)
				{
					tail_grad -= prob * grad(triplet_sample, componet::componet_tail);
				}
				if (relation == false)
				{
					relation_grad -= prob * grad(triplet_sample, componet::componet_relation);
				}

				mat_grad -= prob * grad_matr(triplet_sample, GeometricEmbeddingModel::componet_matr);
				mata_grad -= prob * grad_matr(triplet_sample, GeometricEmbeddingModel::componet_mata);
			}
		}

		if (total_normalizor != 0)
		{
			head_grad /= total_normalizor;
			tail_grad /= total_normalizor;
			relation_grad /= total_normalizor;
			mat_grad /= total_normalizor;
			mata_grad /= total_normalizor;
		}

		head_grad += grad(triplet, GeometricEmbeddingModel::componet_head);
		tail_grad += grad(triplet, GeometricEmbeddingModel::componet_tail);
		relation_grad += grad(triplet, GeometricEmbeddingModel::componet_relation);
		mat_grad += grad_matr(triplet, GeometricEmbeddingModel::componet_matr);
		mata_grad += grad_matr(triplet, GeometricEmbeddingModel::componet_mata);
		
		head -= alpha * head_grad;
		tail -= alpha * tail_grad;
		relation -= alpha * relation_grad;
		mat_relation[triplet.second] -= alpha * mat_grad;
		mat_a[triplet.second] -= alpha * mata_grad;

		head = normalise(head);
		tail = normalise(tail);
		relation = normalise(relation);
	}

	virtual double probability_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet, vec & error )
	{
		error = (mat_relation[triplet.second] * embedding_entity[triplet.first.first]
		+ embedding_relation[triplet.second]
		- mat_relation[triplet.second]* embedding_entity[triplet.first.second]) ;
		
		return - sum(abs(error));
	}
};

class TransGGMPRC
	:public TransGGMP
{
public:
	TransGGMPRC(int dim, double alpha, int sampling_times)
		:TransGGMP(dim, alpha, sampling_times)
	{
		;
	}

public:
	virtual vec grad( const pair<pair<string, string>,string>& triplet, componet part )
	{
		switch(part)
		{
		case GeometricEmbeddingModel::componet_head:
			return embedding_entity[name_entity[triplet.first.first]]
			+ embedding_relation[name_relation[triplet.second]]
			- embedding_entity[name_entity[triplet.first.second]];
			break;
		case GeometricEmbeddingModel::componet_tail:
			return - (embedding_entity[name_entity[triplet.first.first]]
			+ embedding_relation[name_relation[triplet.second]]
			- embedding_entity[name_entity[triplet.first.second]]);
			break;
		case GeometricEmbeddingModel::componet_relation:
			return embedding_entity[name_entity[triplet.first.first]]
			+ embedding_relation[name_relation[triplet.second]]
			- embedding_entity[name_entity[triplet.first.second]];
			break;
		}
	}

	virtual mat grad_matr( const pair<pair<string, string>,string>& triplet, componet part )
	{
		return abs(embedding_entity[name_entity[triplet.first.first]]
			+ embedding_relation[name_relation[triplet.second]]
			- embedding_entity[name_entity[triplet.first.second]])
			* abs(embedding_entity[name_entity[triplet.first.first]]
			+ embedding_relation[name_relation[triplet.second]]
			- embedding_entity[name_entity[triplet.first.second]]).t();
	}

	virtual double prob_triplets( const pair<pair<string, string>,string>& triplet )
	{
		return - norm((mat_relation[name_relation[triplet.second]]
			* embedding_entity[name_entity[triplet.first.first]]
			+ embedding_relation[name_relation[triplet.second]]
			- mat_relation[name_relation[triplet.second]]
			* embedding_entity[name_entity[triplet.first.second]]), 1);
	}

	virtual void train( double alpha )
	{
		for_each(mat_relation.begin(), mat_relation.end(), [=](mat& elem){elem = eye(dim,dim);});
		GeometricEmbeddingModel::train(alpha);
		//for_each(mat_relation.begin(), mat_relation.end(), [=](mat& elem){elem = normalise(elem);});
	}
};

class TransGGMPRM
	:public TransGGMP
{
public:
	TransGGMPRM(int dim, double alpha, int sampling_times)
		:TransGGMP(dim, alpha, sampling_times)
	{
		;
	}

public:
	virtual vec grad( const pair<pair<string, string>,string>& triplet, componet part )
	{
		if (epos < 50)
		{
			switch(part)
			{
			case GeometricEmbeddingModel::componet_head:
				return sign((mat_relation[name_relation[triplet.second]]
				* embedding_entity[name_entity[triplet.first.first]]
				+ embedding_relation[name_relation[triplet.second]]
				- mat_relation[name_relation[triplet.second]]
				* embedding_entity[name_entity[triplet.first.second]]));
				break;
			case GeometricEmbeddingModel::componet_tail:
				return - sign((mat_relation[name_relation[triplet.second]]
				* embedding_entity[name_entity[triplet.first.first]]
				+ embedding_relation[name_relation[triplet.second]]
				- mat_relation[name_relation[triplet.second]]
				* embedding_entity[name_entity[triplet.first.second]]));
				break;
			case GeometricEmbeddingModel::componet_relation:
				return sign((mat_relation[name_relation[triplet.second]]
				* embedding_entity[name_entity[triplet.first.first]]
				+ embedding_relation[name_relation[triplet.second]]
				- mat_relation[name_relation[triplet.second]]
				* embedding_entity[name_entity[triplet.first.second]]));
				break;
			}
		}
		else
		{
			switch(part)
			{
			case GeometricEmbeddingModel::componet_head:
				return mat_relation[name_relation[triplet.second]].t()
					* sign((mat_relation[name_relation[triplet.second]]
				* embedding_entity[name_entity[triplet.first.first]]
				+ embedding_relation[name_relation[triplet.second]]
				- mat_relation[name_relation[triplet.second]]
				* embedding_entity[name_entity[triplet.first.second]]));
				break;
			case GeometricEmbeddingModel::componet_tail:
				return - mat_relation[name_relation[triplet.second]].t()
					* sign((mat_relation[name_relation[triplet.second]]
				* embedding_entity[name_entity[triplet.first.first]]
				+ embedding_relation[name_relation[triplet.second]]
				- mat_relation[name_relation[triplet.second]]
				* embedding_entity[name_entity[triplet.first.second]]));
				break;
			case GeometricEmbeddingModel::componet_relation:
				return sign((mat_relation[name_relation[triplet.second]]
				* embedding_entity[name_entity[triplet.first.first]]
				+ embedding_relation[name_relation[triplet.second]]
				- mat_relation[name_relation[triplet.second]]
				* embedding_entity[name_entity[triplet.first.second]]));
				break;
			}
		}
	}

	virtual mat grad_matr( const pair<pair<string, string>,string>& triplet, componet part )
	{
		if (epos < 50)
			return (mat_relation[name_relation[triplet.second]]
			* embedding_entity[name_entity[triplet.first.first]]
			+ embedding_relation[name_relation[triplet.second]]
			- mat_relation[name_relation[triplet.second]]
			* embedding_entity[name_entity[triplet.first.second]]) 
				* (embedding_entity[name_entity[triplet.first.first]]
			- embedding_entity[name_entity[triplet.first.second]]).t();
		else
			return sign((mat_relation[name_relation[triplet.second]]
			* embedding_entity[name_entity[triplet.first.first]]
			+ embedding_relation[name_relation[triplet.second]]
			- mat_relation[name_relation[triplet.second]]
			* embedding_entity[name_entity[triplet.first.second]]))
				* (embedding_entity[name_entity[triplet.first.first]]
			- embedding_entity[name_entity[triplet.first.second]]).t();


	}

	virtual double prob_triplets( const pair<pair<string, string>,string>& triplet )
	{
		return - sum(abs(
			mat_relation[name_relation[triplet.second]]
		* embedding_entity[name_entity[triplet.first.first]]
		+ embedding_relation[name_relation[triplet.second]]
		- mat_relation[name_relation[triplet.second]]
		* embedding_entity[name_entity[triplet.first.second]]));
	}

	virtual void train( double alpha )
	{
		if (epos > 50)
			for_each(mat_relation.begin(), mat_relation.end(), [=](mat& elem){elem = eye(dim,dim);});
		
		GeometricEmbeddingModel::train(alpha);
		for_each(mat_relation.begin(), mat_relation.end(), [=](mat& elem){elem = normalise(elem);});
	}
};