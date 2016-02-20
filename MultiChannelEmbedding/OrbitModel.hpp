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
	const int	dim;
	const double	alpha;
	const double	training_threshold;

public:
	OrbitModel(	
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
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
		for_each(embedding_entity.begin(), embedding_entity.end(), [=](vec& elem){elem = (2*randu(dim,1)-1)*sqrt(6.0/dim);});

		embedding_relation.resize(count_relation());
		for_each(embedding_relation.begin(), embedding_relation.end(), [=](vec& elem){elem = (2*randu(dim,1)-1)*sqrt(6.0/dim);});
	}

public:
	virtual void draw(const string& filename, const int radius, 
		int id_head, int id_relation)
	{
		mat	record(radius*6.0 + 100, radius*6.0 + 100);
		record.fill(255);
		for(auto i=data_model.rel_tails.at(id_relation).at(id_head).begin();
			i!=data_model.rel_tails.at(id_relation).at(id_head).end();
			++ i)
		{
			if (rand()%100 < 50)
				continue;

			record(50 + radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]), 
				50 + radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1])) = 0;
			record(50 + radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) + 1, 
				50 + radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) + 1) = 0;
			record(50 + radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) + 1, 
				50 + radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) - 1) = 0;
			record(50 + radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) - 1, 
				50 + radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) - 1) = 0;
			record(50 + radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) - 1, 
				50 + radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) + 1) = 0;
			record(50 + radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) + 2, 
				50 + radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) + 2) = 0;
			record(50 + radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) + 2, 
				50 + radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) - 2) = 0;
			record(50 + radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) - 2, 
				50 + radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) - 2) = 0;
			record(50 + radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) - 2, 
				50 + radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) + 2) = 0;
			record(50 + radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) + 3, 
				50 + radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) + 3) = 0;
			record(50 + radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) + 3, 
				50 + radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) - 3) = 0;
			record(50 + radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) - 3, 
				50 + radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) - 3) = 0;
			record(50 + radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) - 3, 
				50 + radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) + 3) = 0;
			record(50 + radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) + 4, 
				50 + radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) + 4) = 0;
			record(50 + radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) + 4, 
				50 + radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) - 4) = 0;
			record(50 + radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) - 4, 
				50 + radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) - 4) = 0;
			record(50 + radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) - 4, 
				50 + radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) + 4) = 0;
		}

		auto changes = make_pair(make_pair(id_head, id_head), id_relation);
		priority_queue<pair<double, int>>	heap_entity;
		for(auto i=0; i<count_entity(); ++i)
		{
			changes.first.first = i;
			heap_entity.push(make_pair(prob_triplets(changes), i));
		}

		for(auto i=0; i<data_model.rel_tails.at(id_relation).at(id_head).size(); ++i)
		{
			heap_entity.pop();
		}

		for(auto j=0; j<100; ++j)
		{
			int t=1;
			while(t--)
				heap_entity.pop();

			int i = heap_entity.top().second;

			record(50 + radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]), 
				50 + radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1])) = 1;
			record(50 + radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) + 1, 
				50 + radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) + 0) = 0;
			record(50 + radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) + 0, 
				50 + radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) - 1) = 0;
			record(50 + radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) - 1, 
				50 + radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) - 0) = 0;
			record(50 + radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) - 0, 
				50 + radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) + 1) = 0;
			record(50 + radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) + 2, 
				50 + radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) + 0) = 0;
			record(50 + radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) + 0, 
				50 + radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) - 2) = 0;
			record(50 + radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) - 2, 
				50 + radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) - 0) = 0;
			record(50 + radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) - 0, 
				50 + radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) + 2) = 0;
			record(50 + radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) + 3, 
				50 + radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) + 0) = 0;
			record(50 + radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) + 0, 
				50 + radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) - 3) = 0;
			record(50 + radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) - 3, 
				50 + radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) - 0) = 0;
			record(50 + radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) - 0, 
				50 + radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) + 3) = 0;
			record(50 + radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) + 4, 
				50 + radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) + 0) = 0;
			record(50 + radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) + 0, 
				50 + radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) - 4) = 0;
			record(50 + radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) - 4, 
				50 + radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) - 0) = 0;
			record(50 + radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] + embedding_relation[id_relation][0]) - 0, 
				50 + radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] + embedding_relation[id_relation][1]) + 4) = 0;

		}

		record.save(filename, pgm_binary);
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

class OrbitE
	:public OrbitModel
{
public:
	OrbitE(	
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold)
		:OrbitModel(dataset, task_type, logging_base_path, 
		dim, alpha, training_threshold)
	{
		logging.record()<<"\t[Name]\tOrbitE";
	}

	virtual double prob_triplets( const pair<pair<int, int>,int>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		double score = 
			fabs(as_scalar(sum(abs(head + relation - tail)) - orbit*orbit));
		return - score;
	}

	virtual void train_triplet( const pair<pair<int, int>,int>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		pair<pair<int, int>,int> triplet_f;
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

class OrbitE2
	:public OrbitModel
{
public:
	OrbitE2(	
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold)
		:OrbitModel(dataset, task_type, logging_base_path, 
		dim, alpha, training_threshold)
	{
		logging.record()<<"\t[Name]\tOrbitE2";
	}

	virtual double prob_triplets( const pair<pair<int, int>,int>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		double score = 
			fabs(as_scalar((head + relation - tail).t()*(head + relation - tail)) - orbit*orbit);

		if (score < 0)
			score = 0;

		return - score;
	}

	virtual void train_triplet( const pair<pair<int, int>,int>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		pair<pair<int, int>,int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];

		double factor = 
			-sign(as_scalar((head + relation - tail).t()*(head + relation - tail)) - orbit*orbit);
		double factor_f = 
			-sign(as_scalar((head_f + relation_f - tail_f).t()*(head_f + relation_f - tail_f)) - orbit*orbit);

		head += alpha * factor * (head + relation - tail);
		relation += alpha * factor * (head + relation - tail);
		tail -= alpha * factor * (head + relation - tail);
		head_f -= alpha * factor_f * (head_f + relation_f - tail_f);
		relation_f -= alpha * factor_f * (head_f + relation_f - tail_f);
		tail_f += alpha * factor_f * (head_f + relation_f - tail);
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
protected:
	vector<vec>	embedding_gap;

public:
	OrbitHyper(	
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold)
		:OrbitModel(dataset, task_type, logging_base_path, 
		dim, alpha, training_threshold)
	{
		logging.record()<<"\t[Name]\tOrbitE HyperPlane";
		embedding_orbit = ones(count_relation());

		embedding_gap.resize(count_relation());
		for(auto i=embedding_gap.begin(); i!=embedding_gap.end(); ++i)
		{
			*i = randu(dim);
		}
	}

public:
	virtual double prob_triplets( const pair<pair<int, int>,int>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		vec& gap = embedding_gap[triplet.second];

		double& orbit = embedding_orbit[triplet.second];
		
		return - fabs(sum(abs(head - relation * head.t() * relation + gap - tail
			+ relation * tail.t() * relation))
			- orbit*orbit);
	}

	virtual void train_triplet( const pair<pair<int, int>,int>& triplet ) 
	{  
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		vec& gap = embedding_gap[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		pair<pair<int, int>,int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];
		vec& gap_f = embedding_gap[triplet_f.second];

		double factor = 
			- sign(sum(abs(head - relation * head.t() * relation + gap - tail + 
			relation * tail.t() * relation)) - orbit*orbit);
		double factor_f = 
			- sign(sum(abs(head_f - relation_f * head_f.t() * relation_f + gap_f - tail_f +
			relation * tail_f.t() * relation)) - orbit*orbit);

		vec factor_vec = sign(head - relation * head.t() * relation + gap - tail + relation * tail.t() * relation);

		head += alpha * factor * (eye(dim, dim)-relation *relation.t()) * factor_vec;
		tail -= alpha * factor * (eye(dim, dim)-relation *relation.t()) * factor_vec;
		relation -= alpha * factor * 
			((eye(dim, dim)* as_scalar((head-tail).t()*relation) + relation*(head-tail).t())) * factor_vec; 
		gap += alpha * factor * factor_vec;

		vec factor_vec_f = sign(head_f - relation_f * head_f.t() * relation_f + gap_f - tail_f + relation_f * tail_f.t() * relation_f);
		head_f -= alpha * factor * (eye(dim, dim)-relation_f *relation_f.t()) * factor_vec_f;
		tail_f += alpha * factor * (eye(dim, dim)-relation_f *relation_f.t()) * factor_vec_f; 
		relation_f += alpha * factor * 
			((eye(dim, dim)* as_scalar((head_f-tail_f).t()*relation_f) + relation_f*(head_f-tail_f).t())) * factor_vec_f; 
		gap_f += alpha * factor * factor_vec_f;

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
		int dim,
		double alpha,
		double training_threshold,
		double ESS_factor)
		:OrbitE(dataset, task_type, logging_base_path, 
		dim, alpha, training_threshold), ESS_factor(ESS_factor)
	{
		logging.record()<<"\t[Name]\tOrbitE_ESS";
	}

public:
	virtual void train_triplet( const pair<pair<int, int>,int>& triplet )
	{
		OrbitE::train_triplet(triplet);
		relation_reg(triplet.second, rand()%count_relation(), ESS_factor);
		//entity_reg(triplet.first.first, rand()%count_entity(), ESS_factor);
		//entity_reg(triplet.first.second, rand()%count_entity(), ESS_factor);
	}
};

class OrbitE_BOX
	:public OrbitModel
{
public:
	OrbitE_BOX(	
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold)
		:OrbitModel(dataset, task_type, logging_base_path, 
		dim, alpha, training_threshold)
	{
		logging.record()<<"\t[Name]\tOrbitE BOX";
	}

	virtual double prob_triplets( const pair<pair<int, int>,int>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		double score = 
			fabs(as_scalar(sum(abs(head + relation - tail)) - orbit*orbit));
		return - score;
	}

	virtual void train_triplet( const pair<pair<int, int>,int>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		pair<pair<int, int>,int> triplet_f;
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

		uword a;
		abs(head + relation - tail).max(a);

		head[a] += alpha * factor * sign(head + relation - tail)[a];
		relation[a] += alpha * factor * sign(head + relation - tail)[a];
		tail[a] -= alpha * factor * sign(head + relation - tail)[a];
		
		uword b;
		abs(head_f + relation_f - tail_f).max(b);

		head_f[b] -= alpha * factor_f * sign(head_f + relation_f - tail_f)[b];
		relation_f[b] -= alpha * factor_f * sign(head_f + relation_f - tail_f)[b];
		tail_f[b] += alpha * factor_f * sign(head_f + relation_f - tail)[b];

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

class MultiLayerPerceptron
{
protected:
	vector<tuple<mat, function<double(const double&)>, function<double(const double&)>>>	weights;
	const vector<int>																		network_architecture;

public:
	MultiLayerPerceptron(const vector<int>& network_architecture,
		function<double(const double&)> fn_active = [](const double x){return (exp(x)-exp(-x))/(exp(x)+exp(-x));}, 
		function<double(const double&)> fn_derv = [](const double x){return 1-x*x;})
		:network_architecture(network_architecture)
	{
		for(auto layer=0; layer<network_architecture.size()-1; ++layer)
		{
			weights.push_back(make_tuple(
				randn(network_architecture[layer], network_architecture[layer+1]), 
				fn_active, fn_derv));
		}
	}

	void add_layer(	const double ndin, 
		const double ndout, 
		function<double(const double&)> fn_active = [](const double x){return 1.0/(1.0+exp(-x));}, 
		function<double(const double&)> fn_derv = [](const double x){return x*(1-x);})
	{
		weights.push_back(make_tuple(randn(ndin, ndout), fn_active, fn_derv));
	}

public:
	virtual double infer(const vec& din) const
	{
		vec dout = din;
		for(auto layer : weights)
		{
			dout = get<0>(layer).t() * dout;
			for_each(dout.begin(), dout.end(), [&](double& elem){elem = get<1>(layer)(elem);});
		}

		return dout[0];
	}

public:
	virtual void train_once(vec& head, vec& tail, const vec& dout, double alpha)
	{
		vector<vec>	hiddens;
		hiddens.push_back(join_cols(head, tail));
		for(auto layer : weights)
		{
			vec hidden_out = get<0>(layer).t() * hiddens.back();
			for_each(hidden_out.begin(), hidden_out.end(), [&](double& elem){elem = get<1>(layer)(elem);});
			hiddens.push_back(hidden_out);
		}

		vec derv = sign(hiddens.back() - dout);
		for(auto layer=weights.size(); layer > 0; --layer)
		{
			for(auto dim_derv=0; dim_derv<derv.n_elem; ++dim_derv)
			{
				derv[dim_derv] *= get<2>(weights[layer-1])(hiddens[layer][dim_derv]);
			}

			get<0>(weights[layer-1]) -= alpha * hiddens[layer-1] * derv.t();
			derv = get<0>(weights[layer-1]) * derv;
		}
		
		head -= alpha * derv.rows(0, head.size()-1);
		tail -= alpha * derv.rows(head.size(), head.size() + tail.size()-1);
	}
};

class OrbitE_Deep
	:public OrbitModel
{
protected:
	vector<MultiLayerPerceptron> mlp;

public:
	OrbitE_Deep(	
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		vector<int>& na,
		function<double(const double&)> fn_active = [](const double x){return 1.0/(1.0+exp(-x));}, 
		function<double(const double&)> fn_derv = [](const double x){return x*(1-x);})
		:OrbitModel(dataset, task_type, logging_base_path, 
		dim, alpha, training_threshold)
	{
		logging.record()<<"\t[Name]\tOrbitE Deep";
		logging.record()<<"\t[Network Structure]\t";
		for(auto i=0; i<na.size(); ++i)
		{
			logging<<na[i]<<'\t';
		}

		for(auto i=0; i<count_relation(); ++i)
		{
			mlp.push_back(MultiLayerPerceptron(na, fn_active, fn_derv));
		}

		embedding_orbit = zeros(count_relation());
	}

	virtual double prob_triplets( const pair<pair<int, int>,int>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		double score = fabs(mlp[triplet.second].infer(join_cols(head, tail)) - orbit*orbit);
		return - score;
	}

	virtual void train_triplet( const pair<pair<int, int>,int>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		pair<pair<int, int>,int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];

		double factor = 
			- sign(mlp[triplet.second].infer(join_cols(head, tail)) - orbit*orbit);
		double factor_f = 
			- sign(mlp[triplet_f.second].infer(join_cols(head_f, tail_f)) - orbit*orbit);

		vec dout(1);
		dout << orbit*orbit;
		
		mlp[triplet.second].train_once(head, tail, dout, alpha);
		mlp[triplet.second].train_once(head_f, tail_f, dout, -alpha);

		//orbit -= alpha * alpha *(factor - factor_f)*orbit;

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


class OrbitE_H
	:public OrbitModel
{
protected:
	vector<vec>	embedding_weights;

public:
	OrbitE_H(	
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold)
		:OrbitModel(dataset, task_type, logging_base_path, 
		dim, alpha, training_threshold)
	{
		logging.record()<<"\t[Name]\tOrbitE H";

		embedding_weights.resize(count_relation());
		for(auto i=embedding_weights.begin(); i!=embedding_weights.end(); ++i)
		{
			*i = (2*randu(dim,1)-1)*sqrt(6.0/dim);
		}

		embedding_orbit.fill(10.0);
	}

	virtual double prob_triplets( const pair<pair<int, int>,int>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		double score = fabs(as_scalar((head + relation).t() * tail) - orbit);
		return - score;
	}

	virtual void train_triplet( const pair<pair<int, int>,int>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		//vec& weight = embedding_weights[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		pair<pair<int, int>,int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];
		//vec& weight_f = embedding_weights[triplet_f.second];

		double factor = - sign(as_scalar((head + relation).t() * tail) - orbit);
		double factor_f = - sign(as_scalar((head_f + relation_f).t() * tail_f) - orbit);

		head += alpha * factor * tail;
		relation += alpha * factor * tail;
		tail += alpha * factor * (head + relation);
		head_f -= alpha * factor_f * tail_f;
		relation_f -= alpha * factor_f * tail_f;
		tail_f -= alpha * factor_f * (head_f + relation_f);

		//orbit -= alpha *(factor - factor_f);

		//if (norm(weight) > 1.0)
		//	weight = normalise(weight);

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

class OrbitE_HD
	:public OrbitModel
{
protected:
	vector<vec>	embedding_weights;

public:
	OrbitE_HD(	
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold)
		:OrbitModel(dataset, task_type, logging_base_path, 
		dim, alpha, training_threshold)
	{
		logging.record()<<"\t[Name]\tOrbitE H";

		embedding_weights.resize(count_relation());
		for(auto i=embedding_weights.begin(); i!=embedding_weights.end(); ++i)
		{
			*i = (2*randu(dim,1)-1)*sqrt(6.0/dim);
		}

		embedding_orbit.fill(10.0);
	}

	virtual double prob_triplets( const pair<pair<int, int>,int>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		vec& weight = embedding_weights[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		double score = fabs(as_scalar((head + relation).t() * (tail + weight)) - orbit);
		return - score;
	}

	virtual void train_triplet( const pair<pair<int, int>,int>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		vec& weight = embedding_weights[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		pair<pair<int, int>,int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];
		vec& weight_f = embedding_weights[triplet_f.second];

		double factor = - sign(as_scalar((head + relation).t() * (tail + weight)) - orbit);
		double factor_f = - sign(as_scalar((head_f + relation_f).t() * (tail_f + weight_f)) - orbit);

		head += alpha * factor * (tail + weight);
		relation += alpha * factor * (tail + weight);
		tail += alpha * factor * (head + relation);
		weight += alpha * factor * (head + relation);

		head_f -= alpha * factor_f * (tail_f + weight_f);
		relation_f -= alpha * factor_f * (tail_f + weight_f);
		tail_f -= alpha * factor_f * (head_f + relation_f);
		weight_f -= alpha * factor_f * (head_f + relation_f);

		orbit -= alpha *(factor - factor_f);

		if (norm(weight) > 1.0)
			weight = normalise(weight);

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

class OrbitE_HDA
	:public OrbitModel
{
protected:
	vector<vec>	embedding_weights;

public:
	OrbitE_HDA(	
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold)
		:OrbitModel(dataset, task_type, logging_base_path, 
		dim, alpha, training_threshold)
	{
		logging.record()<<"\t[Name]\tOrbitE H";

		embedding_weights.resize(count_relation());
		for(auto i=embedding_weights.begin(); i!=embedding_weights.end(); ++i)
		{
			*i = (2*randu(dim,1)-1)*sqrt(6.0/dim);
		}

		for(auto i=embedding_orbit.begin(); i!=embedding_orbit.end(); ++i)
		{
			*i = (2*randu()-1)*sqrt(6.0/dim);
		}

		//embedding_orbit.fill(dim*dim*4);
	}

	virtual double prob_triplets( const pair<pair<int, int>,int>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		vec& weight = embedding_weights[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		double score = fabs(as_scalar(abs(head + relation).t() * abs(tail + weight)) - orbit);
		return - score;
	}

	virtual void train_triplet( const pair<pair<int, int>,int>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		vec& weight = embedding_weights[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		pair<pair<int, int>,int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];
		vec& weight_f = embedding_weights[triplet_f.second];

		double factor = - sign(as_scalar(abs(head + relation).t() * abs(tail + weight)) - orbit);
		double factor_f = - sign(as_scalar(abs(head_f + relation_f).t() * abs(tail_f + weight_f)) - orbit);

		head += alpha * factor * sign(head + relation) % abs(tail + weight);
		relation += alpha * factor * sign(head + relation) % abs(tail + weight);
		tail += alpha * factor * sign(tail + weight) % abs(head + relation);
		weight += alpha * factor * sign(tail + weight) % abs(head + relation);

		head_f -= alpha * factor_f * sign(head_f + relation_f) % abs(tail_f + weight_f);
		relation_f -= alpha * factor_f * sign(head_f + relation_f) % abs(tail_f + weight_f);
		tail_f -= alpha * factor_f * sign(tail_f + weight_f) % abs(head_f + relation_f);
		weight_f -= alpha * factor_f * sign(tail_f + weight_f) % abs(head_f + relation_f);

		orbit -= alpha *(factor - factor_f);

		if (norm(weight) > 1.0)
			weight = normalise(weight);

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

class OrbitE_KS
	:public OrbitModel
{
protected:
	vector<vec>	embedding_weights;
	function<double(const vec& a, const vec& b)>	kernel;	
	function<vec(const vec& a, const vec& b)>		derv_a;
	function<vec(const vec& a, const vec& b)>		derv_b;

public:
	OrbitE_KS(	
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		function<double(const vec& a, const vec& b)> kernel,
		function<vec(const vec& a, const vec& b)> derv_a,
		function<vec(const vec& a, const vec& b)> derv_b)
		:OrbitModel(dataset, task_type, logging_base_path, 
		dim, alpha, training_threshold), 
		kernel(kernel), derv_a(derv_a), derv_b(derv_b)
	{
		logging.record()<<"\t[Name]\tOrbitE H";

		embedding_weights.resize(count_relation());
		for(auto i=embedding_weights.begin(); i!=embedding_weights.end(); ++i)
		{
			*i = (2*randu(dim,1)-1)*sqrt(6.0/dim);
		}

		for(auto i=embedding_orbit.begin(); i!=embedding_orbit.end(); ++i)
		{
			*i = (2*randu()-1)*sqrt(6.0/dim);
		}
	}

	virtual double prob_triplets( const pair<pair<int, int>,int>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		double score = fabs(kernel(head, head) + kernel(tail, tail) + kernel(relation, relation)
			+ 2 * (kernel(head, relation) - kernel(head, tail) - kernel(relation, tail)) - orbit);
		return - score;
	}

	virtual void train_triplet( const pair<pair<int, int>,int>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		pair<pair<int, int>,int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];

		double factor = - sign(kernel(head, head) + kernel(tail, tail) + kernel(relation, relation)
			+ 2 * (kernel(head, relation) - kernel(head, tail) - kernel(relation, tail)) - orbit);
		double factor_f = - sign(kernel(head_f, head_f) + kernel(tail_f, tail_f) + kernel(relation_f, relation_f)
			+ 2 * (kernel(head_f, relation_f) - kernel(head_f, tail_f) - kernel(relation_f, tail_f)) - orbit);

		head += alpha * factor * 
			(derv_a(head, head) + derv_b(head, head) + 2 * derv_a(head, relation) - 2 * derv_a(head, tail));
		relation += alpha * factor * 
			(derv_a(relation, relation) + derv_b(relation, relation) + 2 * derv_b(head, relation) - 2 * derv_a(relation, tail));
		tail += alpha * factor * 
			(derv_a(tail, tail) + derv_b(tail, tail) - 2 * derv_b(head, tail) - 2 * derv_b(relation, tail));

		head_f -= alpha * factor_f * 
			(derv_a(head_f, head_f) + derv_b(head_f, head_f) + 2 * derv_a(head_f, relation_f) - 2 * derv_a(head_f, tail_f));
		relation_f -= alpha * factor_f * 
			(derv_a(relation_f, relation_f) + derv_b(relation_f, relation_f) + 2 * derv_b(head_f, relation_f) - 2 * derv_a(relation_f, tail_f));
		tail_f -= alpha * factor_f * 
			(derv_a(tail_f, tail_f) + derv_b(tail_f, tail_f) - 2 * derv_b(head_f, tail_f) - 2 * derv_b(relation_f, tail_f));

		orbit -= alpha *(factor - factor_f);

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

class OrbitE_KHDA
	:public OrbitModel
{
protected:
	vector<vec>	embedding_weights;
	function<double(const vec& a, const vec& b)>	kernel;	
	function<vec(const vec& a, const vec& b)>		derv_a;
	function<vec(const vec& a, const vec& b)>		derv_b;

public:
	OrbitE_KHDA(	
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		function<double(const vec& a, const vec& b)> kernel,
		function<vec(const vec& a, const vec& b)> derv_a,
		function<vec(const vec& a, const vec& b)> derv_b)
		:OrbitModel(dataset, task_type, logging_base_path, 
		dim, alpha, training_threshold),kernel(kernel), derv_a(derv_a), derv_b(derv_b)
	{
		logging.record()<<"\t[Name]\tOrbitE H";

		embedding_weights.resize(count_relation());
		for(auto i=embedding_weights.begin(); i!=embedding_weights.end(); ++i)
		{
			*i = (2*randu(dim,1)-1)*sqrt(6.0/dim);
		}

		for(auto i=embedding_orbit.begin(); i!=embedding_orbit.end(); ++i)
		{
			*i = (2*randu()-1)*sqrt(6.0/dim);
		}
	}

	virtual double prob_triplets( const pair<pair<int, int>,int>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		vec& weight = embedding_weights[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		double score = fabs(kernel(head, tail) + kernel(relation, tail) + kernel(head, weight)
			+ kernel(relation, weight) - orbit);
		return - score;
	}

	virtual void train_triplet( const pair<pair<int, int>,int>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		vec& weight = embedding_weights[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		pair<pair<int, int>,int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];
		vec& weight_f = embedding_weights[triplet_f.second];

		double factor = - sign(kernel(head, tail) + kernel(relation, tail) + kernel(head, weight)
			+ kernel(relation, weight) - orbit);
		double factor_f = - sign(kernel(head_f, tail_f) + kernel(relation_f, tail_f) + kernel(head_f, weight_f)
			+ kernel(relation_f, weight_f) - orbit);

		head += alpha * factor * (derv_a(head, tail) + derv_a(head, weight));
		relation += alpha * factor * (derv_a(relation, tail) + derv_a(relation, weight));
		tail += alpha * factor * (derv_b(relation, tail) + derv_b(head, tail));
		weight += alpha * factor * (derv_b(head, weight) + derv_b(relation, weight));

		head_f -= alpha * factor_f * (derv_a(head_f, tail_f) + derv_a(head_f, weight_f));
		relation_f -= alpha * factor_f * (derv_a(relation_f, tail_f) + derv_a(relation_f, weight_f));
		tail_f -= alpha * factor_f * (derv_b(relation_f, tail_f) + derv_b(head_f, tail_f));
		weight_f -= alpha * factor_f * (derv_b(head_f, weight_f) + derv_b(relation_f, weight_f));

		orbit -= alpha *(factor - factor_f);

		if (norm(weight) > 1.0)
			weight = normalise(weight);

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

class OrbitE_KHDAN
	:public OrbitModel
{
protected:
	vector<vec>	embedding_weights;
	function<double(const vec& a, const vec& b)>	kernel;	
	function<vec(const vec& a, const vec& b)>		derv_a;
	function<vec(const vec& a, const vec& b)>		derv_b;

public:
	OrbitE_KHDAN(	
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		function<double(const vec& a, const vec& b)> kernel,
		function<vec(const vec& a, const vec& b)> derv_a,
		function<vec(const vec& a, const vec& b)> derv_b)
		:OrbitModel(dataset, task_type, logging_base_path, 
		dim, alpha, training_threshold),kernel(kernel), derv_a(derv_a), derv_b(derv_b)
	{
		logging.record()<<"\t[Name]\tOrbitE H NK";

		embedding_weights.resize(count_relation());
		for(auto i=embedding_weights.begin(); i!=embedding_weights.end(); ++i)
		{
			*i = (2*randu(dim,1)-1)*sqrt(6.0/dim);
		}

		for(auto i=embedding_orbit.begin(); i!=embedding_orbit.end(); ++i)
		{
			*i = (2*randu()-1)*sqrt(6.0/dim);
		}
	}

	virtual double prob_triplets( const pair<pair<int, int>,int>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		vec& weight = embedding_weights[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		double score = fabs(kernel(head + relation, tail + weight) - orbit);
		return - score;
	}

	virtual void train_triplet( const pair<pair<int, int>,int>& triplet ) 
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		vec& weight = embedding_weights[triplet.second];
		double& orbit = embedding_orbit[triplet.second];

		pair<pair<int, int>,int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];
		vec& weight_f = embedding_weights[triplet_f.second];

		double factor = - sign(kernel(head + relation, tail + weight) - orbit);
		double factor_f = - sign(kernel(head_f + relation_f, tail_f + weight_f) - orbit);

		head += alpha * factor * derv_a(head + relation, tail + weight);
		relation += alpha * factor * derv_a(head + relation, tail + weight);
		tail += alpha * factor * derv_b(head + relation, tail + weight);
		weight += alpha * factor * derv_b(head + relation, tail + weight);

		head_f -= alpha * factor_f * derv_a(head_f + relation_f, tail_f + weight_f);
		relation_f -= alpha * factor_f * derv_a(head_f + relation_f, tail_f + weight_f);
		tail_f -= alpha * factor_f * derv_b(head_f + relation_f, tail_f + weight_f);
		weight_f -= alpha * factor_f * derv_b(head_f + relation_f, tail_f + weight_f);

		orbit -= alpha *(factor - factor_f);

		if (norm(weight) > 1.0)
			weight = normalise(weight);

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

auto kernel_poly_2 = [&](const vec& a, const vec& b)
{
	return pow(as_scalar(a.t()*b) - 2, 2);
};

auto derv_a_poly_2 = [&](const vec& a, const vec& b)
{
	return pow(as_scalar(a.t()*b) - 2, 1) * 2 * b;
};

auto derv_b_poly_2 = [&](const vec& a, const vec& b)
{
	return pow(as_scalar(a.t()*b) - 2, 1) * 2 * a;
};