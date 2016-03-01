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

protected:
	const double balance;

public:
	SemanticModel(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		const string& semantic_file,
		int dim,
		double alpha,
		double training_threshold,
		double balance=0.1)
		:TransE(dataset, task_type, logging_base_path, dim, alpha, training_threshold),
		balance(balance)
	{
		logging.record() << "\t[Name]\tSemanticModel.TransE";
		logging.record() << "\t[Dimension]\t" << dim;
		logging.record() << "\t[Learning Rate]\t" << alpha;
		logging.record() << "\t[Training Threshold]\t" << training_threshold;
		logging.record() << "\t[Topic Model]\tLSI";
		logging.record() << "\t[Balance]\t" << balance;

		ifstream fin(semantic_file.c_str(), ios::binary);
		storage_vmat<double>::load(v_semantics, fin);
		fin.close();

		cout << "File Loaded." << endl;
	}

	SemanticModel(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		double balance = 0.1)
		:TransE(dataset, task_type, logging_base_path, dim, alpha, training_threshold),
		balance(balance)
	{
		logging.record() << "\t[Name]\tSemanticModel.TransE";
		logging.record() << "\t[Dimension]\t" << dim;
		logging.record() << "\t[Learning Rate]\t" << alpha;
		logging.record() << "\t[Training Threshold]\t" << training_threshold;
		logging.record() << "\t[Topic Model]\tLSI";
		logging.record() << "\t[Balance]\t" << balance;

		v_semantics.resize(count_entity()+10);
		for (auto& elem : v_semantics)
		{
			elem = randn(dim);
		}
	}

	SemanticModel(
		const Dataset& dataset,
		const string& file_zeroshot,
		const TaskType& task_type,
		const string& logging_base_path,
		const string& semantic_file,
		int dim,
		double alpha,
		double training_threshold,
		double balance = 0.1)
		:TransE(dataset, file_zeroshot, task_type,logging_base_path, dim, alpha, training_threshold),
		balance(balance)
	{
		logging.record() << "\t[Name]\tSemanticModel.TransE";
		logging.record() << "\t[Dimension]\t" << dim;
		logging.record() << "\t[Learning Rate]\t" << alpha;
		logging.record() << "\t[Training Threshold]\t" << training_threshold;
		logging.record() << "\t[Topic Model]\tLSI";
		logging.record() << "\t[Balance]\t" << balance;

		ifstream fin(semantic_file.c_str(), ios::binary);
		storage_vmat<double>::load(v_semantics, fin);
		fin.close();

		cout << "File Loaded." << endl;
	}

	SemanticModel(
		const Dataset& dataset,
		const string& file_zeroshot,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		double balance = 0.1)
		:TransE(dataset, file_zeroshot, task_type, logging_base_path, dim, alpha, training_threshold),
		balance(balance)
	{
		logging.record() << "\t[Name]\tSemanticModel.TransE";
		logging.record() << "\t[Dimension]\t" << dim;
		logging.record() << "\t[Learning Rate]\t" << alpha;
		logging.record() << "\t[Training Threshold]\t" << training_threshold;
		logging.record() << "\t[Topic Model]\tLSI";
		logging.record() << "\t[Balance]\t" << balance;

		v_semantics.resize(count_entity()+10);
		for (auto& elem : v_semantics)
		{
			elem = randn(dim);
		}

		cout << "File Loaded." << endl;
	}

public:
	virtual const vec semantic_composition(const pair<pair<int, int>, int>& triplet) const
	{
		return normalise(v_semantics[triplet.first.first] + v_semantics[triplet.first.second]);
	}

	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet)
	{
		vec semantic = semantic_composition(triplet);

		vec error = embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- embedding_entity[triplet.first.second];

		return	- balance * sum(abs(error - as_scalar(semantic.t()*error)*semantic))
				- sum(abs(error));
	}
	
	virtual void train_triplet(const pair<pair<int, int>, int>& triplet)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];

		pair<pair<int, int>, int> triplet_f;
		data_model.sample_false_triplet_relation(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];

		vec semantic = semantic_composition(triplet);
		vec error = head + relation - tail;
		double projection = as_scalar(semantic.t()*error);
		vec grad = error - projection * (2 - projection) * semantic;
		
		head -= alpha * balance * grad + alpha * sign(error);
		tail += alpha * balance * grad + alpha * sign(error);
		relation -= alpha * balance * grad + alpha * sign(error);

		vec semantic_f = semantic_composition(triplet_f);
		vec error_f = head_f + relation_f - tail_f;
		double projection_f = as_scalar(semantic_f.t()*error_f);
		vec grad_f = error_f - projection_f * (2 - projection_f) * semantic_f;

		head_f += alpha * balance * grad_f + alpha * sign(error_f);
		tail_f -= alpha * balance * grad_f + alpha * sign(error_f);
		relation_f += alpha * balance * grad_f + alpha * sign(error_f);

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

public:
	virtual vec entity_representation(int entity_id) const override
	{
		if (data_model.zeroshot_pointer + 10 < data_model.set_entity.size())
			return v_semantics[entity_id];
		else
			return join_cols(embedding_entity[entity_id], v_semantics[entity_id]);
	}
};

class SemanticModel_Joint
	: public SemanticModel
{
protected:
	vector<vector<string>>		documents;
	map<string, vec>			topic_words;
	vector<string>				words;

protected:
	const double factor;

public:
	SemanticModel_Joint(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		const string& semantic_file_raw,
		int dim,
		double alpha,
		double training_threshold,
		double balance,
		double factor)
		:SemanticModel(dataset, task_type, logging_base_path, dim, alpha, training_threshold, balance),
		factor(factor)
	{
		logging.record() << "\t[Name]\tSemanticModel.Joint";
		logging.record() << "\t[Factor]\t" << factor;

		documents.resize(count_entity() + 10);

		fstream fin(semantic_file_raw.c_str());
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
		}
		fin.close();

		cout << "File Loaded." << endl;
	}

	SemanticModel_Joint(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		const string& semantic_file,
		const string& semantic_file_raw,
		int dim,
		double alpha,
		double training_threshold,
		double balance,
		double factor)
		:SemanticModel(dataset, task_type, logging_base_path, semantic_file, dim, alpha, training_threshold, balance),
		factor(factor)
	{
		logging.record() << "\t[Name]\tSemanticModel.Joint";
		logging.record() << "\t[Factor]\t" << factor;

		documents.resize(count_entity() + 10);

		fstream fin(semantic_file_raw.c_str());
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
		}
		fin.close();

		cout << "File Loaded." << endl;
	}

	SemanticModel_Joint(
		const Dataset& dataset,
		const string& file_zeroshot,
		const TaskType& task_type,
		const string& logging_base_path,
		const string& semantic_file,
		const string& semantic_file_raw,
		int dim,
		double alpha,
		double training_threshold,
		double balance,
		double factor)
		:SemanticModel(dataset, file_zeroshot, task_type, logging_base_path, semantic_file, dim, alpha, training_threshold, balance),
		factor(factor)
	{
		logging.record() << "\t[Name]\tSemanticModel.Joint";
		logging.record() << "\t[Factor]\t" << factor;

		documents.resize(count_entity() + 10);

		fstream fin(semantic_file_raw.c_str());
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
		}
		fin.close();

		cout << "File Loaded." << endl;
	}

	SemanticModel_Joint(
		const Dataset& dataset,
		const string& file_zeroshot,
		const TaskType& task_type,
		const string& logging_base_path,
		const string& semantic_file_raw,
		int dim,
		double alpha,
		double training_threshold,
		double balance,
		double factor)
		:SemanticModel(dataset, file_zeroshot, task_type, logging_base_path, dim, alpha, training_threshold, balance),
		factor(factor)
	{
		logging.record() << "\t[Name]\tSemanticModel.Joint";
		logging.record() << "\t[Factor]\t" << factor;

		documents.resize(count_entity() + 10);

		fstream fin(semantic_file_raw.c_str());
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
		}
		fin.close();

		cout << "File Loaded." << endl;
	}

public:
	virtual void train_topic()
	{
#pragma omp parallel for
		for (auto idoc = documents.begin(); idoc != documents.end(); ++idoc)
		{
			vec& v_doc = v_semantics[idoc - documents.begin()];
			vec v_doc_grad = zeros(dim);

			for (auto iword = idoc->begin(); iword < idoc->end(); ++iword)
			{
				vec& v_word = topic_words[*iword];
				v_doc_grad += alpha * as_scalar(1 - v_doc.t()*v_word) * v_word;
				v_word += alpha * as_scalar(1 - v_doc.t()*v_word) * v_doc;

				vec& v_word_ns = topic_words[words[rand() % words.size()]];
				v_doc_grad += alpha * as_scalar(0 - v_doc.t()*v_word_ns) * v_word_ns;
				v_word_ns += alpha * as_scalar(0 - v_doc.t()*v_word_ns) * v_doc;

				v_word_ns = normalise(v_word_ns);
				v_word = normalise(v_word);
			}

			v_doc += v_doc_grad;
			v_doc = normalise(v_doc);
		}
	}

	virtual void train_triplet(const pair<pair<int, int>, int>& triplet)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		vec& head_sem = v_semantics[triplet.first.first];
		vec& tail_sem = v_semantics[triplet.first.second];

		pair<pair<int, int>, int> triplet_f;
		data_model.sample_false_triplet_relation(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];
		vec& head_sem_f = v_semantics[triplet_f.first.first];
		vec& tail_sem_f = v_semantics[triplet_f.first.second];

		vec semantic = semantic_composition(triplet);
		vec error = head + relation - tail;
		double projection = as_scalar(semantic.t()*error);
		double length = as_scalar(semantic.t()*semantic);
		vec grad = error - projection * (2 - projection) * semantic;

		head -= alpha * balance * grad + alpha * sign(error);
		tail += alpha * balance * grad + alpha * sign(error);
		relation -= alpha * balance * grad + alpha * sign(error);
		head_sem -= alpha * balance * factor * projection * (1 - length) * error;
		tail_sem -= alpha * balance * factor * projection * (1 - length) * error;

		vec semantic_f = semantic_composition(triplet_f);
		vec error_f = head_f + relation_f - tail_f;
		double projection_f = as_scalar(semantic_f.t()*error_f);
		double length_f = as_scalar(semantic_f.t()*semantic_f);
		vec grad_f = error_f - projection_f * (2 - projection_f) * semantic_f;

		head_f += alpha * balance * grad_f + alpha * sign(error_f);
		tail_f -= alpha * balance * grad_f + alpha * sign(error_f);
		relation_f += alpha * balance * grad_f + alpha * sign(error_f);
		head_sem_f += alpha * balance * factor * projection_f * (1 - length_f) * error_f;
		tail_sem_f += alpha * balance * factor * projection_f * (1 - length_f) * error_f;

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

		head_sem = normalise(head_sem);
		tail_sem = normalise(tail_sem);
		head_sem_f = normalise(head_sem_f);
		tail_sem_f = normalise(tail_sem_f);
	}

	virtual void train(bool last_time = false) override
	{
		TransE::train(last_time);
		train_topic();
	}
};

class SemanticModel_ZeroShot
	:public SemanticModel_Joint
{
public:
	SemanticModel_ZeroShot(
		const Dataset& dataset,
		const string& file_zeroshot,
		const TaskType& task_type,
		const string& logging_base_path,
		const string& semantic_file,
		const string& semantic_file_raw,
		int dim,
		double alpha,
		double training_threshold,
		double balance,
		double factor)
		:SemanticModel_Joint(dataset, file_zeroshot, task_type, logging_base_path, semantic_file,
			semantic_file_raw, dim, alpha, training_threshold, balance, factor)
	{
		logging.record() << "\t[Name]\tZeroShot";
	}

public:
	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet) override
	{
		const vec& semantic = semantic_composition(triplet);
		double length = as_scalar(semantic.t()*semantic);
		double p = 2 - (-balance - 1) / length;

		return - p * p * ((p - 1)*(p - 1) + (p - 2)*(p - 2)) * length;
	}
};