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
		logging.record()<<"\t[Name]\tOrbitModel";
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
	virtual double prob_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		double error = (as_scalar((head-relation).t()*(head-relation) 
			+ (tail-relation).t()*(tail-relation)) - orbit*orbit);
		return - fabs(error);
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
			- sign((as_scalar((head-relation).t()*(head-relation) 
			+ (tail-relation).t()*(tail-relation)) - orbit*orbit));
		double factor_f = 
			- sign((as_scalar((head_f-relation_f).t()*(head_f-relation_f) 
			+ (tail_f-relation_f).t()*(tail_f-relation_f)) - orbit*orbit));

		head += alpha * factor * (head-relation);
		tail += alpha * factor * (tail-relation);
		relation -= alpha * factor * (head + tail - 2*relation);
		head_f -= alpha * factor * (head_f-relation_f);
		tail_f -= alpha * factor * (tail_f-relation_f);
		relation_f += alpha * factor * (head_f + tail_f - 2*relation_f);
		orbit -= alpha * (factor - factor_f) * orbit;
	}
};

class OrbitBin
	:public OrbitModel
{
public:
	OrbitBin(	
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		unsigned dim,
		double alpha,
		double training_threshold)
		:OrbitModel(dataset, task_type, logging_base_path, 
		dim, alpha, training_threshold)
	{
		;
	}

	virtual double prob_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		double score = 0;
		for(auto i=0; i<dim; ++i)
		{
			score += fabs(head(i)*head(i) + relation(i)*relation(i) - tail(i)*tail(i));
		}

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

		for(auto i=0; i<dim; ++i)
		{
			double factor = 
				- sign(head(i)*head(i) + relation(i)*relation(i) - tail(i)*tail(i));
			double factor_f = 
				- sign(head_f(i)*head_f(i) + relation_f(i)*relation_f(i) - tail_f(i)*tail_f(i));
			
			head(i) += alpha * factor * head(i);
			relation(i) += alpha * factor * relation(i);
			tail(i) -= alpha * factor * tail(i);
			head_f(i) += alpha * factor * head_f(i);
			relation_f(i) += alpha * factor * relation_f(i);
			tail_f(i) -= alpha * factor * tail_f(i);
		}

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

class OrbitBall
	:public OrbitModel
{
public:
	OrbitBall(	
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		unsigned dim,
		double alpha,
		double training_threshold)
		:OrbitModel(dataset, task_type, logging_base_path, 
		dim, alpha, training_threshold)
	{
		embedding_orbit = ones(count_relation());;
	}

public:
	virtual double prob_triplets( const pair<pair<unsigned, unsigned>,unsigned>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		return - fabs(as_scalar(head.t()*head + tail.t()*tail + relation.t()*relation)
			- orbit * orbit);
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
			- fabs(as_scalar(head.t()*head + tail.t()*tail + relation.t()*relation)
			- orbit * orbit);
		double factor_f = 
			- fabs(as_scalar(head_f.t()*head_f + tail_f.t()*tail_f + relation_f.t()*relation_f)
			- orbit * orbit);

		head += alpha * factor * head;
		tail += alpha * factor * tail;
		relation += alpha * factor * relation;
		head_f -= alpha * factor_f * head_f;
		tail_f -= alpha * factor_f * tail_f;
		relation_f -= alpha * factor_f * relation_f;
	}
};