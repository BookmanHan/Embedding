#pragma once
#include "Import.hpp"
#include "ModelConfig.hpp"
#include "DataModel.hpp"
#include "Model.hpp"

class OrbitModel
	:public Model
{
protected:
	vector<vec>	embedding_entity;
	vector<vec>	embedding_relation;
	vec		embedding_orbit;

protected:
	const unsigned	dim;
	const double	alpha;
	const double	training_threshold;

public:
	OrbitModel(	
		const Dataset& dataset,
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

		embedding_orbit = randu(count_relation());

		embedding_entity.resize(count_entity());
		for_each(embedding_entity.begin(), embedding_entity.end(), [=](vec& elem){elem = randu(dim,1);});

		embedding_relation.resize(count_relation());
		for_each(embedding_relation.begin(), embedding_relation.end(), [=](vec& elem){elem = randu(dim,1);});
	}

public:
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

class OrbitE
	:public OrbitModel
{
public:
	OrbitE(	
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		unsigned dim,
		double alpha,
		double training_threshold)
		:OrbitModel(dataset, task_type, logging_base_path, 
		dim, alpha, training_threshold)
	{
		logging.record()<<"\t[Name]\tOrbitE";
	}

	virtual double prob_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		double score = 
			fabs(as_scalar(sum(abs(head + relation - tail)) - orbit*orbit));
		return - score;
	}

	virtual void train_triplet( const pair<pair<unsigned, unsigned>,unsigned>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		pair<pair<unsigned, unsigned>,unsigned> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];

		double factor = 
			- sign(as_scalar(sum(abs(head + relation - tail)) - orbit*orbit));
		double factor_f = 
			- sign(as_scalar(sum(abs(head_f + relation_f - tail_f)) - orbit*orbit));
			
		head += alpha * factor * sign(head + relation - tail);
		relation += alpha * factor * sign(head + relation - tail);
		tail -= alpha * factor * sign(head + relation - tail);
		head_f -= alpha * factor_f * sign(head_f + relation_f - tail_f);
		relation_f -= alpha * factor_f * sign(head_f + relation_f - tail_f);
		tail_f += alpha * factor_f * sign(head_f + relation_f - tail);
		orbit -= alpha *(factor - factor_f)*orbit;

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
};

class OrbitHyper
	:public OrbitModel
{
public:
	OrbitHyper(	
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		unsigned dim,
		double alpha,
		double training_threshold)
		:OrbitModel(dataset, task_type, logging_base_path, 
		dim, alpha, training_threshold)
	{
		logging.record()<<"\t[Name]\tOrbitE";
		embedding_orbit = ones(count_relation());;
	}

public:
	virtual double prob_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		double& orbit = embedding_orbit[triplet.second];
		
		return - fabs(sum(abs(head - relation * head.t() * relation - tail))
			- orbit*orbit);
	}

	virtual void train_triplet( const pair<pair<unsigned, unsigned>,unsigned>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		pair<pair<unsigned, unsigned>,unsigned> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];

		double factor = 
			- sign(sum(abs(head - relation * head.t() * relation - tail))
			- orbit*orbit);
		double factor_f = 
			- sign(sum(abs(head_f - relation_f * head_f.t() * relation_f - tail_f))
			- orbit*orbit);

		head += alpha * factor * (eye(dim, dim)-relation *relation.t())
			* sign(head - relation * head.t() * relation - tail);
		tail -= alpha * factor * sign(head - relation * head.t() * relation - tail);
		relation -= alpha * factor * (eye(dim, dim)* as_scalar(head.t()*relation) + relation*head.t())
			* sign(head - relation * head.t() * relation - tail);
		head_f -= alpha * factor_f * (eye(dim, dim)-relation_f *relation_f.t())
			* sign(head_f - relation_f * head_f.t() * relation_f - tail_f);
		tail_f += alpha * factor_f * sign(head_f - relation_f * head_f.t() * relation_f - tail_f);
		relation_f += alpha * factor_f * (eye(dim, dim)* as_scalar(head_f.t()*relation_f) + relation_f*head_f.t())
			* sign(head_f - relation_f * head_f.t() * relation - tail_f);
		orbit -= alpha *(factor - factor_f) * orbit;

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
};

class OrbitE_ESS
	:public OrbitE
{
protected:
	double ESS_factor;

public:
	OrbitE_ESS(	
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		unsigned dim,
		double alpha,
		double training_threshold,
		double ESS_factor)
		:OrbitE(dataset, task_type, logging_base_path, 
		dim, alpha, training_threshold), ESS_factor(ESS_factor)
	{
		logging.record()<<"\t[Name]\tOrbitE_ESS";
	}

public:
	virtual void train_triplet( const pair<pair<unsigned, unsigned>,unsigned>& triplet )
	{
		OrbitE::train_triplet(triplet);
		relation_reg(triplet.second, rand()%count_relation(), ESS_factor);
		//entity_reg(triplet.first.first, rand()%count_entity(), ESS_factor);
		//entity_reg(triplet.first.second, rand()%count_entity(), ESS_factor);
	}
};