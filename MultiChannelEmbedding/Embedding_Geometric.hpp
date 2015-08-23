#pragma once
#include "Model.hpp"

#pragma once
#include "Model.hpp"

inline double sign(const double& x)
{
	if (x==0)
		return 0;
	else
		return x>0?+1:-1;
}

double norm_common(const mat& m)
{
	double re = 0;
	for(auto i=m.begin(); i!=m.end(); ++i)
	{
		re += (*i) * (*i);
	}

	return sqrt(re);
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
protected:
	unsigned cnt;

public:
	TransE(int dim, double alpha)
		:GeometricEmbeddingModel(dim, alpha), cnt(0)
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

	void relation_reg(int i, int j, double factor)
	{
			if (i == j)
				return;

			embedding_relation[i] -= factor * sign(as_scalar(embedding_relation[i].t()*embedding_relation[j])) * embedding_relation[j];
			embedding_relation[j] -= factor * sign(as_scalar(embedding_relation[i].t()*embedding_relation[j])) * embedding_relation[i];
	}

	virtual double train_once( const pair<pair<unsigned, unsigned>,unsigned>& triplet, double factor )
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];

		pair<pair<unsigned, unsigned>,unsigned> triplet_f;
		sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > 10)
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

		if (norm(head) > 1.0)
			head = normalise(head);
		
		if (norm(tail) > 1.0)
			tail = normalise(tail);

		if (norm(relation) > 1.0)
			relation = normalise(relation);

		if (norm(head_f) > 1.0)
			head_f = normalise(head_f);

		if (norm(tail_f) > 1.0)
			tail_f = normalise(tail_f);

		//head = normalise(head);
		//tail = normalise(tail);
		//relation = normalise(relation);
		//head_f = normalise(head_f);
		//tail_f = normalise(tail_f);
		//relation_f = normalise(relation_f);

		//relation_reg(triplet.second, rand()%set_relation.size(), factor);
	}
};

class TransN
	:public GeometricEmbeddingModel
{
protected:
	vector<vec>	u_r;

public:
	TransN(int dim, double alpha)
		:GeometricEmbeddingModel(dim, alpha)
	{
		u_r.resize(set_relation.size());
		for_each(u_r.begin(), u_r.end(), [=](vec& elem){elem =  randu(dim,1);});
	}

public:
	virtual double prob_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		vec error = embedding_entity[triplet.first.first]
		+ embedding_relation[triplet.second]
		- embedding_entity[triplet.first.second];

		return - as_scalar(u_r[triplet.second].t() * tanh(abs(error)));
	}

	virtual double train_once( const pair<pair<unsigned, unsigned>,unsigned>& triplet, double factor )
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		vec error = embedding_entity[triplet.first.first]
		+ embedding_relation[triplet.second]
		- embedding_entity[triplet.first.second];
		vec& u = u_r[triplet.second];

		pair<pair<unsigned, unsigned>,unsigned> triplet_f;
		sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > 2)
			return 0;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];
		vec error_f = embedding_entity[triplet_f.first.first]
		+ embedding_relation[triplet_f.second]
		- embedding_entity[triplet_f.first.second];

		head -= alpha *  (ones(dim, 1) - tanh(abs(error))%tanh(abs(error))) % sign(error) % u;
		tail += alpha *  (ones(dim, 1) - tanh(abs(error))%tanh(abs(error))) % sign(error) % u;
		relation -= alpha *  (ones(dim, 1) - tanh(abs(error))%tanh(abs(error))) % sign(error) % u;
		head_f += alpha *  (ones(dim, 1) - tanh(abs(error_f))%tanh(abs(error_f))) % sign(error_f) % u;
		tail_f -= alpha *  (ones(dim, 1) - tanh(abs(error_f))%tanh(abs(error_f))) % sign(error_f) % u;
		relation_f += alpha *  (ones(dim, 1) - tanh(abs(error_f))%tanh(abs(error_f))) % sign(error_f) % u;
		u_r[triplet.second] -= alpha * (tanh(abs(error)) - tanh(abs(error_f)));

		//head = normalise(head);
		//tail = normalise(tail);
		//relation = normalise(relation);
		//head_f = normalise(head_f);
		//tail_f = normalise(tail_f);
		//relation_f = normalise(relation_f);
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
	unsigned cnt;

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

		return -as_scalar(abs(error).t()*mat_r[triplet.second]*abs(error));
	}

	virtual double probability_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		vec error = embedding_entity[triplet.first.first]
		+ embedding_relation[triplet.second]
		- embedding_entity[triplet.first.second];

		return - sum(abs(error));
	}

	void relation_reg(int i, int j, double factor)
	{
		if (i == j)
			return;

		embedding_relation[i] -= factor * sign(as_scalar(embedding_relation[i].t()*embedding_relation[j])) * embedding_relation[j];
		embedding_relation[j] -= factor * sign(as_scalar(embedding_relation[i].t()*embedding_relation[j])) * embedding_relation[i];
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

		relation_reg(triplet.second, rand()%set_relation.size(), 0.1);
	}

	virtual void train( double alpha )
	{
		++ cnt;
		TransE::train(alpha);
		
		if (epos == 3000)
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
			//for_each(mat_r.begin(), mat_r.end(), [=](mat& elem){elem = normalise(elem);});
		}
	}
};

class TransG
	:public GeometricEmbeddingModel
{
protected:
	int	n_cluster;
	vector<vector<vec>>		embedding_clusters;
	vector<vector<mat>>		embedding_metric;
	vector<vec>						weights_clusters;
	double							garma;
	double							lambda;
	unsigned						time_limits;
	double							sparse_factor;
	const	bool					single_or_total;

public:
	TransG(int dim, int cluster,  double garma, double alpha, double sparse_factor, bool sot=true)
		:GeometricEmbeddingModel(dim, alpha), n_cluster(cluster), 
		garma(garma), sparse_factor(sparse_factor), single_or_total(true)
	 {
		embedding_clusters.resize(set_relation.size());
		for(auto &elem_vec : embedding_clusters)
		{
			elem_vec.resize(n_cluster);
			for_each(elem_vec.begin(), elem_vec.end(), [=](vec& elem){elem = randu(dim,1);});
		}

		embedding_metric.resize(set_relation.size());
		for(auto &elem_vec : embedding_metric)
		{
			elem_vec.resize(n_cluster);
			for_each(elem_vec.begin(), elem_vec.end(), [=](mat& elem){elem = eye(dim,dim);});
		}

		weights_clusters.resize(set_relation.size());
		for(auto & elem_vec : weights_clusters)
		{
			elem_vec.resize(n_cluster);
			for(auto & elem : elem_vec)
			{
				elem = 1.0/n_cluster;
			}
		}
	}

public:
	virtual double prob_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		if (single_or_total == false)
			return probability_triplets(triplet);

		double	mixed_prob = 1e-8;
		for(int c=0; c<n_cluster; ++c)
		{
			vec error_c = embedding_entity[triplet.first.first] + embedding_clusters[triplet.second][c]
			- embedding_entity[triplet.first.second];
			mixed_prob = max(mixed_prob, fabs(weights_clusters[triplet.second][c]) * exp(-sum(abs(error_c))));
		}

		return mixed_prob;
	}
	
	virtual double probability_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		double	mixed_prob = 1e-8;
		for(int c=0; c<n_cluster; ++c)
		{
			vec error_c = embedding_entity[triplet.first.first] + embedding_clusters[triplet.second][c]
			- embedding_entity[triplet.first.second];
			mixed_prob += fabs(weights_clusters[triplet.second][c]) * exp(-sum(abs(error_c)));
		}

		return mixed_prob;
	}

	virtual void train_cluster_once(	const pair<pair<unsigned, unsigned>,unsigned>& triplet, 
																		const pair<pair<unsigned, unsigned>,unsigned>& triplet_f, 
																		int cluster, double prob_true, double prob_false, double factor)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_clusters[triplet.second][cluster];
		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_clusters[triplet_f.second][cluster];

		double prob_local_true =  exp(-sum(abs(head + relation - tail)));
		double prob_local_false =  exp(-sum(abs(head_f + relation_f - tail_f)));
		
		weights_clusters[triplet.second][cluster] += 
			alpha /prob_true * prob_local_true * sign(weights_clusters[triplet.second][cluster]);
		weights_clusters[triplet_f.second][cluster] -= 
			alpha /prob_false * prob_local_false  * sign(weights_clusters[triplet_f.second][cluster]);

		weights_clusters[triplet.second][cluster] -= 
			alpha * sign(weights_clusters[triplet.second][cluster]);

		head -= alpha * sign(head + relation - tail) 
			* prob_local_true/prob_true * fabs(weights_clusters[triplet.second][cluster]);
		tail += alpha * sign(head + relation - tail)
			* prob_local_true/prob_true * fabs(weights_clusters[triplet.second][cluster]);
		relation -= alpha * sign(head + relation - tail)
			* prob_local_true/prob_true * fabs(weights_clusters[triplet.second][cluster]);
		head_f += alpha * sign(head_f + relation_f - tail_f)
			* prob_local_false/prob_false * fabs(weights_clusters[triplet.second][cluster]);
		tail_f -= alpha * sign(head_f + relation_f - tail_f)
			* prob_local_false/prob_false  * fabs(weights_clusters[triplet.second][cluster]);
		relation_f += alpha * sign(head_f + relation_f - tail_f)
			* prob_local_false/prob_false * fabs(weights_clusters[triplet.second][cluster]);

		relation = normalise(relation);
		relation_f = normalise(relation_f);
	}

	virtual double train_once( const pair<pair<unsigned, unsigned>,unsigned>& triplet, double factor )
	{
		pair<pair<unsigned, unsigned>,unsigned> triplet_f;
		sample_false_triplet(triplet, triplet_f);

		double prob_true = (probability_triplets(triplet));
		double prob_false = (probability_triplets(triplet_f));

		if (prob_true/prob_false > garma)
			return 0;

		for(int c=0; c<n_cluster; ++c)
		{
			train_cluster_once(triplet, triplet_f, c, prob_true, prob_false, factor);
		}

		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];

		head = normalise(head);
		tail = normalise(tail);
		head_f = normalise(head_f);
		tail_f = normalise(tail_f);
		
		//weights_clusters[triplet.second] = normalise(weights_clusters[triplet.second]);
	}

	virtual void train( double alpha )
	{
		EmbeddingModel::train(alpha);

		if (epos == 1000)
		for(auto i=weights_clusters.begin(); i!=weights_clusters.end(); ++i)
		{
			//cout<<number_relation[i-weights_clusters.begin()]<<" : ";
			int num = 0;
			double max_elem = abs(*i).max();
			double norma = norm(*i);
			for_each(i->begin(), i->end(), 
				[&](double & elem)
			{
				if (abs(elem) > 0.05 * max_elem)
					++ num;
				else
					elem = 0;
			});
			//cout<<num;
			//cout<<endl;
		}
	}
}; 

class TransGP
	:public GeometricEmbeddingModel
{
protected:
	int	n_cluster;
	vector<vector<vec>>		embedding_clusters;
	vector<vector<mat>>		embedding_metric;
	vector<vec>						weights_clusters;
	double							garma;
	double							lambda;
	unsigned						time_limits;
	double							sparse_factor;
	const	bool					single_or_total;
	mat									head_prior;
	mat									tail_prior;

public:
	TransGP(int dim, int cluster,  double garma, double alpha, double sparse_factor, bool sot=true)
		:GeometricEmbeddingModel(dim, alpha), n_cluster(cluster), 
		garma(garma), sparse_factor(sparse_factor), single_or_total(true)
	{
		embedding_clusters.resize(set_relation.size());
		for(auto &elem_vec : embedding_clusters)
		{
			elem_vec.resize(n_cluster);
			for_each(elem_vec.begin(), elem_vec.end(), [=](vec& elem){elem = randu(dim,1);});
		}

		embedding_metric.resize(set_relation.size());
		for(auto &elem_vec : embedding_metric)
		{
			elem_vec.resize(n_cluster);
			for_each(elem_vec.begin(), elem_vec.end(), [=](mat& elem){elem = eye(dim,dim);});
		}

		weights_clusters.resize(set_relation.size());
		for(auto & elem_vec : weights_clusters)
		{
			elem_vec.resize(n_cluster);
			for(auto & elem : elem_vec)
			{
				elem = 1.0/n_cluster;
			}
		}

		head_prior.resize(set_relation.size(), set_entity.size());
		tail_prior.resize(set_relation.size(), set_entity.size());
	}

public:
	virtual double prob_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		double	mixed_prob = 1e-8;
		if (single_or_total == false)
		{
			for(int c=0; c<n_cluster; ++c)
			{
				vec error_c = embedding_entity[triplet.first.first] + embedding_clusters[triplet.second][c]
				- embedding_entity[triplet.first.second];
				mixed_prob += fabs(weights_clusters[triplet.second][c]) * exp(-sum(abs(error_c)));
			}
		}
		else
		{
			for(int c=0; c<n_cluster; ++c)
			{
				vec error_c = embedding_entity[triplet.first.first] + embedding_clusters[triplet.second][c]
						- embedding_entity[triplet.first.second];
						mixed_prob = max(mixed_prob, fabs(weights_clusters[triplet.second][c]) * exp(-sum(abs(error_c))));
			}
		}

		return mixed_prob 
			* exp(head_prior(triplet.second, triplet.first.first))
			* exp(tail_prior(triplet.second, triplet.first.second));
	}

	virtual double probability_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		double	mixed_prob = 1e-8;
		for(int c=0; c<n_cluster; ++c)
		{
			vec error_c = embedding_entity[triplet.first.first] + embedding_clusters[triplet.second][c]
			- embedding_entity[triplet.first.second];
			mixed_prob += fabs(weights_clusters[triplet.second][c]) * exp(-sum(abs(error_c)));
		}

		return mixed_prob ;
	}

	virtual void train_cluster_once(	const pair<pair<unsigned, unsigned>,unsigned>& triplet, 
		const pair<pair<unsigned, unsigned>,unsigned>& triplet_f, 
		int cluster, double prob_true, double prob_false, double factor)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_clusters[triplet.second][cluster];
		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_clusters[triplet_f.second][cluster];

		double prob_local_true =  exp(-sum(abs(head + relation - tail)));
		double prob_local_false =  exp(-sum(abs(head_f + relation_f - tail_f)));

		weights_clusters[triplet.second][cluster] += 
			alpha /prob_true * prob_local_true * sign(weights_clusters[triplet.second][cluster]);
		weights_clusters[triplet_f.second][cluster] -= 
			alpha /prob_false * prob_local_false  * sign(weights_clusters[triplet_f.second][cluster]);

		weights_clusters[triplet.second][cluster] -= 
			alpha * sign(weights_clusters[triplet.second][cluster]);

		head -= alpha * sign(head + relation - tail) 
			* prob_local_true/prob_true * fabs(weights_clusters[triplet.second][cluster]);
		tail += alpha * sign(head + relation - tail)
			* prob_local_true/prob_true * fabs(weights_clusters[triplet.second][cluster]);
		relation -= alpha * sign(head + relation - tail)
			* prob_local_true/prob_true * fabs(weights_clusters[triplet.second][cluster]);
		head_f += alpha * sign(head_f + relation_f - tail_f)
			* prob_local_false/prob_false * fabs(weights_clusters[triplet.second][cluster]);
		tail_f -= alpha * sign(head_f + relation_f - tail_f)
			* prob_local_false/prob_false  * fabs(weights_clusters[triplet.second][cluster]);
		relation_f += alpha * sign(head_f + relation_f - tail_f)
			* prob_local_false/prob_false * fabs(weights_clusters[triplet.second][cluster]);

		relation = normalise(relation);
		relation_f = normalise(relation_f);
	}

	virtual double train_once( const pair<pair<unsigned, unsigned>,unsigned>& triplet, double factor )
	{
		pair<pair<unsigned, unsigned>,unsigned> triplet_f;
		sample_false_triplet(triplet, triplet_f);

		double prob_true = (probability_triplets(triplet));
		double prob_false = (probability_triplets(triplet_f));

		if (prob_true/prob_false > garma)
			return 0;

		for(int c=0; c<n_cluster; ++c)
		{
			train_cluster_once(triplet, triplet_f, c, prob_true, prob_false, factor);
		}

		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];

		head_prior(triplet.second, triplet.first.first) += alpha;
		tail_prior(triplet.second, triplet.first.second) += alpha;
		
		head_prior(triplet_f.second, triplet_f.first.first) -= alpha;
		tail_prior(triplet_f.second, triplet_f.first.second) -= alpha;

		head = normalise(head);
		tail = normalise(tail);
		head_f = normalise(head_f);
		tail_f = normalise(tail_f);
	}

	virtual void train( double alpha )
	{
		EmbeddingModel::train(alpha);
		head_prior = normalise(head_prior, 2, 1);
		tail_prior = normalise(tail_prior, 2, 1);
	}
}; 

class TransGPA
	:public GeometricEmbeddingModel
{
protected:
	int	n_cluster;
	vector<vector<vec>>		embedding_clusters;
	vector<vector<mat>>		embedding_metric;
	vector<vec>						weights_clusters;
	double							garma;
	double							lambda;
	unsigned						time_limits;
	double							sparse_factor;
	const	bool					single_or_total;
	mat									head_prior;
	mat									tail_prior;

public:
	TransGPA(	int dim, int cluster,  double garma, double alpha, double sparse_factor, 
						int time_limit, double lamda = 1, bool sot=true)
		:GeometricEmbeddingModel(dim, alpha), n_cluster(cluster),  lambda(lamda),
		garma(garma), sparse_factor(sparse_factor), single_or_total(true), time_limits(time_limit)
	{
		embedding_clusters.resize(set_relation.size());
		for(auto &elem_vec : embedding_clusters)
		{
			elem_vec.resize(n_cluster);
			for_each(elem_vec.begin(), elem_vec.end(), [=](vec& elem){elem = randu(dim,1);});
		}

		embedding_metric.resize(set_relation.size());
		for(auto &elem_vec : embedding_metric)
		{
			elem_vec.resize(n_cluster);
			for_each(elem_vec.begin(), elem_vec.end(), [=](mat& elem){elem = eye(dim,dim);});
		}

		weights_clusters.resize(set_relation.size());
		for(auto & elem_vec : weights_clusters)
		{
			elem_vec.resize(n_cluster);
			for(auto & elem : elem_vec)
			{
				elem = 1.0/n_cluster;
			}
		}

		head_prior.resize(set_relation.size(), set_entity.size());
		tail_prior.resize(set_relation.size(), set_entity.size());
	}

public:
	virtual double prob_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		double	mixed_prob = 1e-8;
		if (single_or_total == false)
		{
			for(int c=0; c<n_cluster; ++c)
			{
				vec error_c = embedding_entity[triplet.first.first] + embedding_clusters[triplet.second][c]
				- embedding_entity[triplet.first.second];
				mixed_prob += fabs(weights_clusters[triplet.second][c]) 
					* exp(-as_scalar(abs(error_c).t()*embedding_metric[triplet.second][c]*abs(error_c)));
			}
		}
		else
		{
			for(int c=0; c<n_cluster; ++c)
			{
				vec error_c = embedding_entity[triplet.first.first] + embedding_clusters[triplet.second][c]
				- embedding_entity[triplet.first.second];
				mixed_prob = max(mixed_prob, 
					fabs(weights_clusters[triplet.second][c])
					* exp(-as_scalar(abs(error_c).t()*embedding_metric[triplet.second][c]*abs(error_c))));
			}
		}

		return mixed_prob;
	}

	virtual double probability_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		double	mixed_prob = 1e-8;
		for(int c=0; c<n_cluster; ++c)
		{
			vec error_c = embedding_entity[triplet.first.first] + embedding_clusters[triplet.second][c]
			- embedding_entity[triplet.first.second];
			mixed_prob += fabs(weights_clusters[triplet.second][c]) * exp(-sum(abs(error_c)));
		}

		return mixed_prob ;
	}

	virtual void train_cluster_once(	const pair<pair<unsigned, unsigned>,unsigned>& triplet, 
		const pair<pair<unsigned, unsigned>,unsigned>& triplet_f, 
		int cluster, double prob_true, double prob_false, double factor)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_clusters[triplet.second][cluster];
		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_clusters[triplet_f.second][cluster];

		double prob_local_true =  exp(-sum(abs(head + relation - tail)));
		double prob_local_false =  exp(-sum(abs(head_f + relation_f - tail_f)));

		weights_clusters[triplet.second][cluster] += 
			alpha /prob_true * prob_local_true * sign(weights_clusters[triplet.second][cluster]);
		weights_clusters[triplet_f.second][cluster] -= 
			alpha /prob_false * prob_local_false  * sign(weights_clusters[triplet_f.second][cluster]);

		weights_clusters[triplet.second][cluster] -= 
			alpha * sign(weights_clusters[triplet.second][cluster]);

		head -= alpha * sign(head + relation - tail) 
			* prob_local_true/prob_true * fabs(weights_clusters[triplet.second][cluster]);
		tail += alpha * sign(head + relation - tail)
			* prob_local_true/prob_true * fabs(weights_clusters[triplet.second][cluster]);
		relation -= alpha * sign(head + relation - tail)
			* prob_local_true/prob_true * fabs(weights_clusters[triplet.second][cluster]);
		head_f += alpha * sign(head_f + relation_f - tail_f)
			* prob_local_false/prob_false * fabs(weights_clusters[triplet.second][cluster]);
		tail_f -= alpha * sign(head_f + relation_f - tail_f)
			* prob_local_false/prob_false  * fabs(weights_clusters[triplet.second][cluster]);
		relation_f += alpha * sign(head_f + relation_f - tail_f)
			* prob_local_false/prob_false * fabs(weights_clusters[triplet.second][cluster]);

		relation = normalise(relation);
		relation_f = normalise(relation_f);
	}

	virtual double train_once( const pair<pair<unsigned, unsigned>,unsigned>& triplet, double factor )
	{
		pair<pair<unsigned, unsigned>,unsigned> triplet_f;
		sample_false_triplet(triplet, triplet_f);

		double prob_true = (probability_triplets(triplet));
		double prob_false = (probability_triplets(triplet_f));

		if (prob_true/prob_false > garma)
			return 0;

		for(int c=0; c<n_cluster; ++c)
		{
			train_cluster_once(triplet, triplet_f, c, prob_true, prob_false, factor);
		}

		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];

		head = normalise(head);
		tail = normalise(tail);
		head_f = normalise(head_f);
		tail_f = normalise(tail_f);
	}

	virtual void train( double alpha )
{
		EmbeddingModel::train(alpha);
		
		if (epos == time_limits || epos%1000 == 0)
		{
			for(auto &elem_vec : embedding_metric)
			{
				for_each(elem_vec.begin(), elem_vec.end(), [=](mat& elem){elem = eye(dim,dim);});
			}

			for(auto i=i_data_train.begin(); i!=i_data_train.end(); ++i)
			{
				auto& triplet = *i;
				pair<pair<unsigned, unsigned>,unsigned> triplet_f;
				sample_false_triplet(triplet, triplet_f);

				vec& head = embedding_entity[triplet.first.first];
				vec& tail = embedding_entity[triplet.first.second];
				vec& head_f = embedding_entity[triplet_f.first.first];
				vec& tail_f = embedding_entity[triplet_f.first.second];

				double prob_true = probability_triplets(triplet);
				double prob_false = probability_triplets(triplet_f);

				for(auto c=0; c<n_cluster; ++c)
				{
					vec& relation = embedding_clusters[triplet.second][c];
					vec& relation_f = embedding_clusters[triplet_f.second][c];

					embedding_metric[triplet.second][c] -=  
						//exp(-sum(head + relation + tail))/ prob_true * 
						abs(head + relation - tail) * abs(head + relation - tail).t()
						- //exp(-sum(head_f + relation_f + tail_f)) *
						abs(head_f + relation_f - tail_f) * abs(head_f + relation_f - tail_f).t();
				}
			}

			for(auto r=0; r<set_relation.size(); ++r)
			{
				for(auto c=0; c<n_cluster; ++c)
				{
					embedding_metric[r][c] =
						lambda * normalise(embedding_metric[r][c]);
				}
			}
		}
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
		if (epos <= 500)
			return pre_train_once(triplet, factor);

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


class TransGH
	:public GeometricEmbeddingModel
{
protected:
	unsigned int sampling_times;
	vec			 error;
	vector<vec>	 hyper;

public:
	TransGH(int dim, double alpha, int sampling_times =1)
		:GeometricEmbeddingModel(dim, alpha), 
		sampling_times(sampling_times),
		error(dim, 1)
	{
		hyper.resize(set_relation.size());
		for_each(hyper.begin(), hyper.end(), [=](vec& elem){elem = randu(dim, 1);});
	}

public:
	virtual double prob_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		vec error = embedding_entity[triplet.first.first] 
			- as_scalar(hyper[triplet.second].t()* embedding_entity[triplet.first.first]) * hyper[triplet.second]
			+ embedding_relation[triplet.second]
			+ embedding_entity[triplet.first.second] 
			- as_scalar(hyper[triplet.second].t()* embedding_entity[triplet.first.second])
				* hyper[triplet.second];

		return - sum(abs(error));
	}

	virtual double probability_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet)
	{
		return prob_triplets(triplet);
	}

	virtual double pre_train_once( const pair<pair<unsigned, unsigned>,unsigned>& triplet, double factor )
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];

		pair<pair<unsigned, unsigned>,unsigned> triplet_f;
		sample_false_triplet(triplet, triplet_f);

		if (probability_triplets(triplet) -  probability_triplets(triplet_f) > 1)
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

public:
	virtual vec grad(const pair<pair<unsigned, unsigned>,unsigned>& triplet, componet part)
	{
		vec error = embedding_entity[triplet.first.first] 
		- as_scalar(hyper[triplet.second].t()* embedding_entity[triplet.first.first]) * hyper[triplet.second]
		+ embedding_relation[triplet.second]
		+ embedding_entity[triplet.first.second] 
		- as_scalar(hyper[triplet.second].t()* embedding_entity[triplet.first.second])
			* hyper[triplet.second];

		switch(part)
		{
		case GeometricEmbeddingModel::componet_head:
			return (eye(dim,dim) - hyper[triplet.second] * hyper[triplet.second].t()) * sign(error);
			break;
		case GeometricEmbeddingModel::componet_tail:
			return - (eye(dim,dim) - hyper[triplet.second] * hyper[triplet.second].t()) * sign(error);
			break;
		case GeometricEmbeddingModel::componet_relation:
			return sign(error);
			break;
		}
	}

public:
	virtual double train_once( const pair<pair<unsigned, unsigned>,unsigned>& triplet, double factor )
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		vec& hyper = embedding_relation[triplet.second];

		vec head_grad(dim, 1, fill::zeros);
		vec tail_grad(dim, 1, fill::zeros);
		vec relation_grad(dim, 1, fill::zeros);
		vec hyper_grad(dim, 1, fill::zeros);

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
					hyper_grad -= prob * grad(triplet_sample, componet::componet_hyper);
				}
			}
		}

		if (total_normalizor != 0)
		{
			head_grad /= total_normalizor;
			tail_grad /= total_normalizor;
			relation_grad /= total_normalizor;
			hyper_grad /= total_normalizor;
		}

		head_grad += grad(triplet, GeometricEmbeddingModel::componet_head);
		tail_grad += grad(triplet, GeometricEmbeddingModel::componet_tail);
		relation_grad += grad(triplet, GeometricEmbeddingModel::componet_relation);
		hyper_grad += grad(triplet, GeometricEmbeddingModel::componet_hyper);

		head -= factor * head_grad;
		tail -= factor * tail_grad;
		relation -= factor * relation_grad;
		hyper -= factor * hyper_grad;

		head = normalise(head);
		tail = normalise(tail);
		relation = normalise(relation);
		hyper = normalise(hyper);
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

		if (epos == 1000)
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