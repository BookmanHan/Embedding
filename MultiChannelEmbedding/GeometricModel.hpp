#pragma once
#include "Import.hpp"
#include "ModelConfig.hpp"
#include "DataModel.hpp"
#include "Model.hpp"

class TransE
	:public Model
{
protected:
	vector<vec>	embedding_entity;
	vector<vec>	embedding_relation;

public:
	const unsigned	dim;
	const double	alpha;
	const double	training_threshold;

public:
	TransE(	const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		unsigned dim,
		double alpha,
		double training_threshold)
		:Model(dataset, task_type, logging_base_path), 
		dim(dim), alpha(alpha), training_threshold(training_threshold)
	{
		logging.record()<<"\t[Dimension]\t"<<dim;
		logging.record()<<"\t[Learning Rate]\t"<<alpha;
		logging.record()<<"\t[Training Threshold]\t"<<training_threshold;

		embedding_entity.resize(count_entity());
		for_each(embedding_entity.begin(), embedding_entity.end(), [=](vec& elem){elem = randu(dim,1);});

		embedding_relation.resize(count_relation());
		for_each(embedding_relation.begin(), embedding_relation.end(), [=](vec& elem){elem = randu(dim,1);});
	}

	virtual void draw(const string& filename, const unsigned radius, const unsigned id_relation) const
	{
		mat	record(radius*6.0 +4, radius*6.0 + 4);
		record.fill(255);
		for(auto i=data_model.data_train.begin(); i!=data_model.data_train.end(); ++i)
		{
			if (i->second == id_relation)
			{
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0] - embedding_relation[i->second][0]), 
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1] - embedding_relation[i->second][1])) = 0;
			}
		}

		string relation_name = data_model.relation_id_to_name[id_relation];
		record.save(filename + replace_all(relation_name, "/", "_") + ".ppm", pgm_binary);
	}

	virtual void draw(const string& filename, const unsigned radius, 
		unsigned int id_head, unsigned int id_relation) const
	{
		mat	record(radius*6.0 +4, radius*6.0 + 4);
		record.fill(255);
		for(auto i=data_model.rel_heads.at(id_relation).at(id_head).begin();
			i!=data_model.rel_heads.at(id_relation).at(id_head).end();
			++ i)
		{
			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]), 
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1])) = 0;
			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) + 1, 
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) + 1) = 0;
			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) + 1, 
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) - 1) = 0;
			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) - 1, 
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) - 1) = 0;
			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) - 1, 
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) + 1) = 0;
			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) + 2, 
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) + 2) = 0;
			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) + 2, 
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) - 2) = 0;
			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) - 2, 
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) - 2) = 0;
			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) - 2, 
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) + 2) = 0;
		}

		for(auto j=0; j<25; ++j)
		{
			int i = rand()%count_entity();
			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]), 
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1])) = 255;
			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) + 1, 
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) + 1) = 0;
			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) + 1, 
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) - 1) = 0;
			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) - 1, 
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) - 1) = 0;
			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) - 1, 
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) + 1) = 0;
			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) + 1, 
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) + 2) = 0;
			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) + 1, 
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) - 2) = 0;
			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) - 1, 
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) - 2) = 0;
			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) - 1, 
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) + 2) = 0;
		}

		record.save("D:\\1.ppm", pgm_binary);
	}

	virtual double prob_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		vec error = embedding_entity[triplet.first.first]
		+ embedding_relation[triplet.second]
		- embedding_entity[triplet.first.second];

		return - sum(abs(error));
	}

	virtual void train_triplet( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];

		pair<pair<unsigned, unsigned>,unsigned> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

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
	}

	virtual void relation_reg(int i, int j, double factor)
	{
		if (i == j)
			return;

		embedding_relation[i] -= factor * sign(as_scalar(embedding_relation[i].t()*embedding_relation[j])) * embedding_relation[j];
		embedding_relation[j] -= factor * sign(as_scalar(embedding_relation[i].t()*embedding_relation[j])) * embedding_relation[i];
	}

	virtual void entity_reg(int i, int j, double factor)
	{
		if (i == j)
			return;

		embedding_entity[i] -= factor * sign(as_scalar(embedding_entity[i].t()*embedding_entity[j])) * embedding_entity[j];
		embedding_entity[j] -= factor * sign(as_scalar(embedding_entity[i].t()*embedding_entity[j])) * embedding_entity[i];
	}
};

class TransE_ESS
	:public TransE
{
protected:
	const double ESS_factor;

public:
	TransE_ESS(	
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		unsigned dim,
		double alpha,
		double training_threshold,
		double ESS_factor)
		:TransE(dataset, task_type, logging_base_path, dim, alpha, training_threshold), 
		ESS_factor(ESS_factor)
	{
		logging.record()<<"\t[ESS Factor]\t"<<ESS_factor;
	}

	virtual void train_triplet( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		TransE::train_triplet(triplet);
		relation_reg(triplet.second, rand()%count_relation(), ESS_factor);
		entity_reg(triplet.first.first, rand()%count_entity(), ESS_factor);
		entity_reg(triplet.first.second, rand()%count_entity(), ESS_factor);
	}
};

class TransA
	:public TransE
{
protected:
	vector<mat>	mat_r;

public:
	TransA(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		unsigned dim,
		double alpha,
		double training_threshold)
		:TransE(dataset, task_type, logging_base_path, dim, alpha, training_threshold)
	{
		mat_r.resize(count_relation());
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

	virtual double training_prob_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		vec error = embedding_entity[triplet.first.first]
		+ embedding_relation[triplet.second]
		- embedding_entity[triplet.first.second];

		return - sum(abs(error));
	}

	virtual void train_triplet( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];

		pair<pair<unsigned, unsigned>,unsigned> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (training_prob_triplets(triplet) - training_prob_triplets(triplet_f) > training_threshold)
			return;

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

	virtual void train( bool last_time)
	{
		TransE::train(alpha);

		if ( last_time )
		{
			for_each(mat_r.begin(), mat_r.end(), [&](mat& m){ m = eye(dim,dim);});
			for(auto i=data_model.data_train.begin(); i!=data_model.data_train.end(); ++i)
			{
				auto triplet = *i;

				vec& head = embedding_entity[triplet.first.first];
				vec& tail = embedding_entity[triplet.first.second];
				vec& relation = embedding_relation[triplet.second];

				pair<pair<unsigned, unsigned>,unsigned> triplet_f;
				data_model.sample_false_triplet(triplet, triplet_f);

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

class TransA_ESS
	:public TransA
{
protected:
	const double ESS_factor;

public:
	TransA_ESS(	
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		unsigned dim,
		double alpha,
		double training_threshold,
		double ESS_factor)
		:TransA(dataset, task_type, logging_base_path, dim, alpha, training_threshold), 
		ESS_factor(ESS_factor)
	{
		logging.record()<<"\t[ESS Factor]\t"<<ESS_factor;
	}

	virtual void train_triplet( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		TransA::train_triplet(triplet);
		entity_reg(triplet.first.first, rand()%count_entity(), ESS_factor);
		entity_reg(triplet.first.second, rand()%count_entity(), ESS_factor);
	}
};

class TransM
	:public Model
{
protected:
	vector<vec>			embedding_entity;
	vector<vector<vec>>		embedding_clusters;
	vector<vec>			weights_clusters;
	
protected:
	const int			n_cluster;
	const double			alpha;
	const double			sparse_factor;
	const bool			single_or_total;
	const double			training_threshold;
	const unsigned			dim;
	const bool			be_weight_normalized;

public:
	TransM(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		unsigned dim,
		double alpha,
		double training_threshold,
		int n_cluster,  
		double sparse_factor, 
		bool sot = true,
		bool be_weight_normalized = false )
		:Model(dataset, task_type, logging_base_path), dim(dim), alpha(alpha),
		training_threshold(training_threshold), n_cluster(n_cluster), sparse_factor(sparse_factor),
		single_or_total(sot), be_weight_normalized(be_weight_normalized)
	{
		logging.record()<<"\t[Name]\tTransM";
		logging.record()<<"\t[Dimension]\t"<<dim;
		logging.record()<<"\t[Learning Rate]\t"<<alpha;
		logging.record()<<"\t[Training Threshold]\t"<<training_threshold;
		logging.record()<<"\t[Cluster Counts]\t"<<n_cluster;
		logging.record()<<"\t[Sparsity Factor]\t"<<sparse_factor;
		
		if (be_weight_normalized)
			logging.record()<<"\t[Weight Normalized]\tTrue";
		else
			logging.record()<<"\t[Weight Normalized]\tFalse";
		
		if (sot)
			logging.record()<<"\t[Single or Total]\tTrue";
		else
			logging.record()<<"\t[Single or Total]\tFalse";

		embedding_entity.resize(count_entity());
		for_each(embedding_entity.begin(), embedding_entity.end(), [=](vec& elem){elem = randu(dim,1);});

		embedding_clusters.resize(count_relation());
		for(auto &elem_vec : embedding_clusters)
		{
			elem_vec.resize(n_cluster);
			for_each(elem_vec.begin(), elem_vec.end(), [=](vec& elem){elem = randu(dim,1);});
		}

		weights_clusters.resize(count_relation());
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
			return training_prob_triplets(triplet);

		double	mixed_prob = 1e-8;
		for(int c=0; c<n_cluster; ++c)
		{
			vec error_c = embedding_entity[triplet.first.first] + embedding_clusters[triplet.second][c]
			- embedding_entity[triplet.first.second];
			mixed_prob = max(mixed_prob, fabs(weights_clusters[triplet.second][c]) * exp(-sum(abs(error_c))));
		}

		return mixed_prob;
	}

	virtual double training_prob_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
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

	virtual void train_cluster_once(	
		const pair<pair<unsigned, unsigned>,unsigned>& triplet, 
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

		if (norm(relation) > 1.0)
			relation = normalise(relation);

		if (norm(relation_f) > 1.0)
			relation_f = normalise(relation_f);
	}

	virtual void train_triplet( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		pair<pair<unsigned, unsigned>,unsigned> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		double prob_true = (training_prob_triplets(triplet));
		double prob_false = (training_prob_triplets(triplet_f));

		if (prob_true/prob_false > training_threshold)
			return;

		for(int c=0; c<n_cluster; ++c)
		{
			train_cluster_once(triplet, triplet_f, c, prob_true, prob_false, alpha);
		}

		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];

		if (norm(head) > 1.0)
			head = normalise(head);

		if (norm(tail) > 1.0)
			tail = normalise(tail);

		if (norm(head_f) > 1.0)
			head_f = normalise(head_f);

		if (norm(tail_f) > 1.0)
			tail_f = normalise(tail_f);

		if (be_weight_normalized)
			weights_clusters[triplet.second] = normalise(weights_clusters[triplet.second]);
	}
}; 