#pragma once
#include "Import.hpp"
#include "ModelConfig.hpp"
#include "DataModel.hpp"
#include "Model.hpp"
#include "GeometricModel.hpp"
#include <boost/format.hpp>  
#include <boost/tokenizer.hpp>  
#include <boost/algorithm/string.hpp>  

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

		v_semantics.resize(count_entity()+10);
		for (auto i = v_semantics.begin(); i != v_semantics.end(); ++i)
		{
			i->resize(dim);
		}

		fstream fin(semantic_file);
		while (!fin.eof())
		{
			string	name;
			fin >> name;

			int	pos = data_model.entity_name_to_id.find(name)->second;
			for (auto i = 0; i < dim; ++i)
			{
				fin >> v_semantics[pos][i];
			}
		}
	}

public:
	virtual const vec semantic_composition(const pair<pair<int, int>, int>& triplet) const
	{
		return (v_semantics[triplet.first.first] + v_semantics[triplet.first.second]);
	}

	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet)
	{
		vec semantic = semantic_composition(triplet);

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

		pair<pair<int, int>, int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];

		vec semantic = semantic_composition(triplet);
		vec error = head + relation - tail;
		double projection = as_scalar(semantic.t()*error);
		vec grad = error - projection * (2 - projection) * semantic;
		
		head -= alpha * grad;
		tail += alpha * grad;
		relation -= alpha * grad;

		vec semantic_f = semantic_composition(triplet_f);
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

class SemanticModel_Joint
	: public TransE
{
protected:
	vector<vector<string>>		documents;
	vector<vec>					topic_documents;
	map<string, vec>			topic_words;
	vector<string>				words;

protected:
	const double balance;

public:
	SemanticModel_Joint(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		const string& semantic_file,
		int dim,
		double alpha,
		double training_threshold,
		double balance)
		:TransE(dataset, task_type, logging_base_path, dim, alpha, training_threshold),
		balance(balance)
	{
		logging.record() << "\t[Name]\tSemanticModel Joint";
		logging.record() << "\t[Dimension]\t" << dim;
		logging.record() << "\t[Learning Rate]\t" << alpha;
		logging.record() << "\t[Training Threshold]\t" << training_threshold;
		logging.record() << "\t[Topic Model]\tLSI";

		topic_documents.resize(count_entity()+10);
		documents.resize(count_entity()+10);

		for (auto i = topic_documents.begin(); i != topic_documents.end(); ++i)
		{
			*i = randu(dim);
		}

		fstream fin(semantic_file);
		boost::char_separator<char> sep(" \t \"\',.\\?!#%@");
		while (!fin.eof())
		{
			string strin;
			getline(fin, strin);
			boost::tokenizer<boost::char_separator<char>>	token(strin, sep);

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

			documents[data_model.entity_name_to_id.find(entity_name)->second] = entity_description;
			topic_documents.push_back(randu(dim));
		}
	}

public:
	virtual const vec semantic_composition(const pair<pair<int, int>, int>& triplet) const
	{
		return normalise(topic_documents[triplet.first.first] + topic_documents[triplet.first.second]);
	}

	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet)
	{
		vec semantic = semantic_composition(triplet);

		vec error = embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- embedding_entity[triplet.first.second];

		return -sum(pow(error - as_scalar(semantic.t()*error)*semantic, 2));
	}

	virtual void train_topic()
	{
#pragma omp parallel for
		for (auto idoc = documents.begin(); idoc != documents.end(); ++idoc)
		{
			vec& v_doc = topic_documents[idoc - documents.begin()];
			vec v_doc_grad = zeros(dim);

			for (auto iword = idoc->begin(); iword != idoc->end(); ++iword)
			{
				vec& v_word = topic_words[*iword];
				v_doc_grad += alpha * as_scalar(1 - v_doc.t()*v_word) * v_word;
				v_word += alpha * as_scalar(1 - v_doc.t()*v_word) * v_doc;

				vec& v_word_ns = topic_words[words[rand() % words.size()]];
				v_doc_grad -= alpha * balance * as_scalar(1 - v_doc.t()*v_word_ns) * v_word_ns;
				v_word_ns -= alpha * balance * as_scalar(1 - v_doc.t()*v_word_ns) * v_doc;

				v_word_ns = max(v_word_ns, zeros(dim));
				v_word = max(v_word, zeros(dim));

				v_word_ns = normalise(v_word_ns);
				v_word = normalise(v_word);
			}

			v_doc += v_doc_grad;
			v_doc = max(v_doc, zeros(dim));
			v_doc = normalise(v_doc);
		}
	}

	virtual void train_triplet(const pair<pair<int, int>, int>& triplet)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		vec& head_sem = topic_documents[triplet.first.first];
		vec& tail_sem = topic_documents[triplet.first.second];

		pair<pair<int, int>, int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];
		vec& head_sem_f = topic_documents[triplet_f.first.first];
		vec& tail_sem_f = topic_documents[triplet_f.first.second];

		vec semantic = semantic_composition(triplet);
		vec error = head + relation - tail;
		double projection = as_scalar(semantic.t()*error);
		double length = as_scalar(semantic.t()*semantic);
		vec grad = error - projection * (2 - projection) * semantic;

		head -= alpha * grad;
		tail += alpha * grad;
		relation -= alpha * grad;
		head_sem -= alpha * projection * (1 - length) * error;
		tail_sem -= alpha * projection * (1 - length) * error;

		vec semantic_f = semantic_composition(triplet_f);
		vec error_f = head_f + relation_f - tail_f;
		double projection_f = as_scalar(semantic_f.t()*error_f);
		double length_f = as_scalar(semantic_f.t()*semantic_f);
		vec grad_f = error_f - projection_f * (2 - projection_f) * semantic_f;

		head_f += alpha * grad_f;
		tail_f -= alpha * grad_f;
		relation_f += alpha * grad_f;
		head_sem_f += alpha * projection_f * (1 - length_f) * error_f;
		tail_sem_f += alpha * projection_f * (1 - length_f) * error_f;

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

		head_sem_f = normalise(head_sem_f);
		tail_sem_f = normalise(tail_sem_f);
		head_sem = normalise(head_sem);
		tail_sem = normalise(tail_sem);
	}

	virtual void train(bool last_time = false) override
	{
		if (epos < 10)
			train_topic();
		else
			TransE::train(last_time);

		if (epos % 10 == 0)
			train_topic();
	}
};