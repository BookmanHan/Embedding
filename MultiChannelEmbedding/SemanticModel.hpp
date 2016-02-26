#pragma once
#include "Import.hpp"
#include "ModelConfig.hpp"
#include "DataModel.hpp"
#include "Model.hpp"
#include "GeometricModel.hpp"

class SemanticModel
	: public TransE
{
protected:
	vector<vec>	v_semantics;

public:
	SemanticModel(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		const string& semantic_file,
		int dim,
		double alpha,
		double training_threshold)
		:TransE(dataset, task_type, logging_base_path, dim, alpha, training_threshold)
	{
		logging.record() << "\t[Name]\tSemanticModel.TransE";
		logging.record() << "\t[Dimension]\t" << dim;
		logging.record() << "\t[Learning Rate]\t" << alpha;
		logging.record() << "\t[Training Threshold]\t" << training_threshold;
		logging.record() << "\t[Topic Model]\tLSI";

		v_semantics.resize(count_entity());
		for (auto i = v_semantics.begin(); i != v_semantics.end(); ++i)
		{
			i->resize(dim);
		}

		fstream fin(semantic_file);
		while (!fin.eof())
		{
			string	name;
			int		pos = data_model.entity_name_to_id.find(name)->second;

			fin >> name;
			for (auto i = 0; i < dim; ++i)
			{
				fin >> v_semantics[pos][i];
			}
		}
	}

public:
	virtual const vec semantic_composition(const vec& head, const vec& tail, const int rel) const
	{
		return normalise(head + tail);
	}

	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet)
	{
		vec semantic = semantic_composition(
			v_semantics[triplet.first.first],
			v_semantics[triplet.first.second], triplet.second);

		vec error = embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- embedding_entity[triplet.first.second];

		return -sum(abs(error - as_scalar(semantic.t()*error)*semantic));
	}
	
	virtual void train_triplet(const pair<pair<int, int>, int>& triplet)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		vec& head_semantic = v_semantics[triplet.first.first];
		vec& tail_semantic = v_semantics[triplet.first.second];

		pair<pair<int, int>, int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];
		vec& head_semantic_f = v_semantics[triplet_f.first.first];
		vec& tail_semantic_f = v_semantics[triplet_f.first.second];

		vec semantic = semantic_composition(head_semantic, tail_semantic, triplet.second);
		vec error = head + relation - tail;
		double projection = as_scalar(semantic.t()*error);
		vec grad = error - projection * (2 - projection) * semantic;
		
		head -= alpha * grad;
		tail += alpha * grad;
		relation -= alpha * grad;

		vec semantic_f = semantic_composition(head_semantic_f, tail_semantic_f, triplet_f.second);
		vec error_f = head_f + relation_f - tail_f;
		double projection_f = as_scalar(semantic_f.t()*error_f);
		vec grad_f = error_f - projection_f * (2 - projection_f) * semantic_f;

		head_f += alpha * grad_f;
		tail_f -= alpha * grad_f;
		relation_f += alpha * grad_f;

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

class SemanticModel_F
	:public SemanticModel
{
protected:
	vector<vec>	relation_specific;

public:
	SemanticModel_F(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		const string& semantic_file,
		int dim,
		double alpha,
		double training_threshold)
		:SemanticModel(dataset, task_type, logging_base_path, semantic_file,
			dim, alpha, training_threshold)
	{
		logging.record() << "\t[Name]\tSemanticModel.TransE.LR";

		relation_specific.resize(count_relation());
		for (auto i = relation_specific.begin(); i != relation_specific.end(); ++i)
		{
			*i = randu(dim);
		}
	}

public:
	virtual const vec semantic_composition(const vec& head, const vec& tail, const int rel) const override
	{
		return normalise(relation_specific[rel] % (head + tail));
	}

	virtual void train_triplet(const pair<pair<int, int>, int>& triplet)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		vec& rspecific = relation_specific[triplet.second];
		vec& head_semantic = v_semantics[triplet.first.first];
		vec& tail_semantic = v_semantics[triplet.first.second];

		pair<pair<int, int>, int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];
		vec& rspecific_f = relation_specific[triplet_f.second];
		vec& head_semantic_f = v_semantics[triplet_f.first.first];
		vec& tail_semantic_f = v_semantics[triplet_f.first.second];

		vec semantic = semantic_composition(head_semantic, tail_semantic, triplet.second);
		vec error = head + relation - tail;
		double projection = as_scalar(semantic.t()*error);
		vec grad = error - projection * (2 - projection) * semantic;

		head -= alpha * grad;
		tail += alpha * grad;
		relation -= alpha * grad;
		rspecific += alpha * as_scalar(semantic.t() * (head_semantic + tail_semantic)) * error;

		vec semantic_f = semantic_composition(head_semantic_f, tail_semantic_f, triplet_f.second);
		vec error_f = head_f + relation_f - tail_f;
		double projection_f = as_scalar(semantic_f.t()*error_f);
		vec grad_f = error_f - projection_f * (2 - projection_f) * semantic_f;

		head_f += alpha * grad_f;
		tail_f -= alpha * grad_f;
		relation_f += alpha * grad_f;
		rspecific += alpha * as_scalar(semantic_f.t() * (head_semantic_f + tail_semantic_f)) * error_f;

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

		if (norm(rspecific) > 1.0)
			rspecific = normalise(rspecific);
	}
};