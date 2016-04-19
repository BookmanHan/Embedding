#pragma once
#include "Import.hpp"
#include "ModelConfig.hpp"
#include "DataModel.hpp"
#include "Model.hpp"
#include "Storage.hpp"
#include <boost/format.hpp>  
#include <boost/tokenizer.hpp>  
#include <boost/algorithm/string.hpp>  
#include <cctype>

class LatentModel
	:public Model
{
protected:
	field<vec>		embedding_head;
	field<vec>		embedding_tail;
	vec				embedding_topic;
	const int		n_topic;

protected:
	field<vec>		embedding_head_comp;
	field<vec>		embedding_tail_comp;
	vec				embedding_topic_comp;
	vector<vec>		embedding_relation;

public:
	LatentModel(		
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int topic)
		:Model(dataset, task_type, logging_base_path), n_topic(topic)
	{
			embedding_head = field<vec>(count_relation(), count_entity());
			for(auto i=embedding_head.begin(); i!=embedding_head.end(); ++i)
			{
				*i = randu(n_topic,1) * sqrt(6.0/n_topic);
			}

			embedding_tail = field<vec>(count_relation(), count_entity());
			for(auto i=embedding_tail.begin(); i!=embedding_tail.end(); ++i)
			{
				*i = randu(n_topic,1) * sqrt(6.0/n_topic);
			}

			embedding_topic.resize(n_topic);
			embedding_topic.fill(1.0/n_topic);
			normalize_to_prob();

			embedding_head_comp = field<vec>(count_relation(), count_entity());
			embedding_tail_comp = field<vec>(count_relation(), count_entity());

			embedding_topic_comp.resize(n_topic);
			embedding_relation.resize(count_relation());

			for(auto i=embedding_relation.begin(); i!=embedding_relation.end(); ++i)
			{
				*i = zeros(n_topic, 1);
			}

			for(auto j=0; j<count_relation(); ++j)
			{
				for(auto i=0; i<count_entity(); ++i)
				{
					embedding_relation[j] += embedding_head(j, i) + embedding_tail(j, i);
				}		
			}

			for(auto i=embedding_relation.begin(); i!=embedding_relation.end(); ++i)
			{
				*i /= sum(*i) + 1e-100;
			}
	}

public:
	void normalize_to_prob()
	{
		for(auto i=0; i<n_topic; ++i)
		{
			double total_sum = 0;
			for(auto item=embedding_head.begin(); item!=embedding_head.end(); ++item)
			{
				total_sum += (*item)(i);
			}

			if (total_sum > 1e-100)
			{
				for(auto item=embedding_head.begin(); item!=embedding_head.end(); ++item)
				{
					(*item)(i) /= total_sum;
				}
			}
		}

		for(auto i=0; i<n_topic; ++i)
		{
			double total_sum = 0;
			for(auto item=embedding_tail.begin(); item!=embedding_tail.end(); ++item)
			{
				total_sum += (*item)(i);
			}

			if (total_sum > 1e-100)
			{
				for(auto item=embedding_tail.begin(); item!=embedding_tail.end(); ++item)
				{
					(*item)(i) /= total_sum;
				}
			}
		}

		for(auto item=embedding_head.begin(); item!=embedding_head.end(); ++item)
		{
			*item += 0.01/(count_entity()*count_relation());
		}

		for(auto item=embedding_tail.begin(); item!=embedding_tail.end(); ++item)
		{
			*item += 0.01/(count_entity()*count_relation());
		}

		embedding_topic /= sum(embedding_topic);
	}

	virtual double prob_triplets(const pair<pair<int, int>,int>& triplet)
	{
		return sum(
			embedding_head(triplet.second, triplet.first.first) 
			% embedding_tail(triplet.second, triplet.first.second)
			% embedding_topic / embedding_relation[triplet.second]) + 1e-100;
	}

	virtual void train_triplet(const pair<pair<int, int>,int>& triplet)
	{
		embedding_head_comp(triplet.second, triplet.first.first) += 
			embedding_head(triplet.second, triplet.first.first) 
			% embedding_topic / prob_triplets(triplet);
		embedding_tail_comp(triplet.second, triplet.first.second) += 
			embedding_tail(triplet.second, triplet.first.second)
			% embedding_topic / prob_triplets(triplet);	
		embedding_topic_comp += embedding_topic / prob_triplets(triplet);
	}

	virtual void train(bool last_time = false)
	{
		for(auto i=embedding_head_comp.begin(); i!=embedding_head_comp.end(); ++i)
		{
			*i = zeros(n_topic, 1);
		}

		for(auto i=embedding_tail_comp.begin(); i!=embedding_tail_comp.end(); ++i)
		{
			*i = zeros(n_topic, 1);
		}

		embedding_topic_comp = zeros(n_topic, 1);

		Model::train(last_time);

		embedding_head = embedding_head_comp;
		embedding_tail = embedding_tail_comp;
		embedding_topic = embedding_topic_comp;

		normalize_to_prob();

		for(auto i=embedding_relation.begin(); i!=embedding_relation.end(); ++i)
		{
			*i = zeros(n_topic, 1);
		}

		for(auto j=0; j<count_relation(); ++j)
		{
			for(auto i=0; i<count_entity(); ++i)
			{
				embedding_relation[j] += embedding_tail(j, i) + embedding_head(j, i);
			}		
		}

		for(auto i=embedding_relation.begin(); i!=embedding_relation.end(); ++i)
		{
			*i /= sum(*i) + 1e-100;
		}

		cout<<embedding_topic.t();
	}
};

class PropergationModel
	:public Model
{
protected:
	vector<vec>		relation_in;
	vector<vec>		relation_out;
	vector<vec>		entity;
	const int		dim;
	const double	alpha;
	const double	training_threshold;

public:
	PropergationModel(		
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
		logging.record()<<"\t[Name]\tPropergation Model";

		relation_in.resize(count_relation());
		for_each(relation_in.begin(), relation_in.end(), 
			[&](vec & elem){elem = (2*randu(dim,1)-1)*sqrt(6.0/dim);});

		relation_out.resize(count_relation());
		for_each(relation_out.begin(), relation_out.end(), 
			[&](vec & elem){elem = ones(dim);  (2 * randu(dim, 1) - 1)*sqrt(6.0 / dim); });

		entity.resize(count_entity());
		for_each(entity.begin(), entity.end(), 
			[&](vec & elem){elem = (2*randu(dim,1)-1)*sqrt(6.0/dim);});
	}

	virtual double prob_triplets(const pair<pair<int, int>,int>& triplet)
	{
		double score = sum(abs(
			entity[triplet.first.first] % relation_in[triplet.second] 
			- entity[triplet.first.second] % relation_out[triplet.second]));
		
		return - score;
	}

	virtual void train_triplet(const pair<pair<int, int>,int>& triplet)
	{
		pair<pair<int, int>,int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

		vec factor_vec = - sign(entity[triplet.first.first] % relation_in[triplet.second] 
		- entity[triplet.first.second] % relation_out[triplet.second]);

		entity[triplet.first.first] += 
			alpha * factor_vec % relation_in[triplet.second];
		relation_in[triplet.second] +=
			alpha * factor_vec % entity[triplet.first.first];
		entity[triplet.first.second] -= 
			alpha * factor_vec % relation_out[triplet.second];
		//relation_out[triplet.second] -=
		//	alpha * factor_vec % entity[triplet.first.second];

		factor_vec = - sign(entity[triplet_f.first.first] % relation_in[triplet_f.second] 
		- entity[triplet_f.first.second] % relation_out[triplet_f.second]);

		entity[triplet_f.first.first] -= 
			alpha * factor_vec % relation_in[triplet_f.second];
		relation_in[triplet_f.second] -=
			alpha * factor_vec % entity[triplet_f.first.first];
		entity[triplet_f.first.second] += 
			alpha * factor_vec % relation_out[triplet_f.second];
		//relation_out[triplet_f.second] +=
		//	alpha * factor_vec % entity[triplet_f.first.second];

		if (norm_L2(entity[triplet.first.first]) > 1)
			entity[triplet.first.first] = normalise(entity[triplet.first.first]);
		if (norm_L2(entity[triplet.first.second]) > 1)
			entity[triplet.first.second] = normalise(entity[triplet.first.second]);
		if (norm_L2(entity[triplet_f.first.first]) > 1)
			entity[triplet_f.first.first] = normalise(entity[triplet_f.first.first]);
		if (norm_L2(entity[triplet_f.first.second]) > 1)
			entity[triplet_f.first.second] = normalise(entity[triplet_f.first.second]);

		relation_in[triplet.second] = normalise(relation_in[triplet.second]);
		relation_out[triplet.second] = normalise(relation_out[triplet.second]);
	}
};

class FactorEKL
	:public Model
{
protected:
	vector<vec>	embedding_entity;
	vector<vec>	embedding_relation_head;
	vector<vec>	embedding_relation_tail;
	vec			prob_head;
	vec			prob_tail;

protected:
	vec			acc_prob_head;
	vec			acc_prob_tail;

public:
	const double	margin;
	const double	alpha;
	const double	smoothing;
	const int		dim;

public:
	FactorEKL(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		double smoothing)
		:Model(dataset, task_type, logging_base_path),
		dim(dim), alpha(alpha), margin(training_threshold), smoothing(smoothing)
	{
		logging.record() << "\t[Name]\tFactorE";
		logging.record() << "\t[Dimension]\t" << dim;
		logging.record() << "\t[Learning Rate]\t" << alpha;
		logging.record() << "\t[Training Threshold]\t" << training_threshold;
		logging.record() << "\t[Smoothing]\t" << smoothing;

		embedding_entity.resize(count_entity());
		for_each(embedding_entity.begin(), embedding_entity.end(),
			[=](vec& elem){elem = normalise(randu(dim), 1); });

		embedding_relation_head.resize(count_relation());
		for_each(embedding_relation_head.begin(), embedding_relation_head.end(),
			[=](vec& elem){elem = normalise(randu(dim), 1); });

		embedding_relation_tail.resize(count_relation());
		for_each(embedding_relation_tail.begin(), embedding_relation_tail.end(),
			[=](vec& elem){elem = normalise(randu(dim), 1); });

		prob_head = normalise(ones(dim));
		prob_tail = normalise(ones(dim));
	}

public:
	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet) override
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation_head = embedding_relation_head[triplet.second];
		vec& relation_tail = embedding_relation_tail[triplet.second];

		vec head_feature = max(head % relation_head, smoothing * ones(dim));
		vec tail_feature = max(tail % relation_tail, smoothing * ones(dim));

		return - sum(head_feature % log(head_feature / tail_feature));
	}

	virtual void train_derv(const pair<pair<int, int>, int>& triplet, const double alpha) = 0;

public:
	virtual void save(const string& filename) override
	{
		ofstream fout(filename.c_str(), ios::binary);
		storage_vmat<double>::save(embedding_entity, fout);
		storage_vmat<double>::save(embedding_relation_head, fout);
		storage_vmat<double>::save(embedding_relation_tail, fout);
		fout.close();
	}

	virtual void load(const string& filename) override
	{
		ifstream fin(filename.c_str(), ios::binary);
		storage_vmat<double>::load(embedding_entity, fin);
		storage_vmat<double>::load(embedding_relation_head, fin);
		storage_vmat<double>::load(embedding_relation_tail, fin);
		fin.close();
	}

public:
	virtual vec entity_representation(int entity_id) const
	{
		return embedding_entity[entity_id];
	}
};

class FactorE
	:public FactorEKL
{
protected:
	const double sigma;

public:
	FactorE(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		double sigma)
		:FactorEKL(dataset, task_type, logging_base_path, dim, alpha, training_threshold, 0.0),
		sigma(sigma)
	{
		logging.record() << "\t[Sigma]\t" << sigma;
	}

public:
	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet) override
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation_head = embedding_relation_head[triplet.second];
		vec& relation_tail = embedding_relation_tail[triplet.second];

		vec head_feature = head % relation_head;
		vec tail_feature = tail % relation_tail;

		return log(sum(head_feature % tail_feature)) * sigma
			-sum(abs(head_feature - tail_feature));
	}

	virtual void train_derv(const pair<pair<int, int>, int>& triplet, const double alpha) override
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation_head = embedding_relation_head[triplet.second];
		vec& relation_tail = embedding_relation_tail[triplet.second];

		vec head_feature = head % relation_head;
		vec tail_feature = tail % relation_tail;
		vec feature = head_feature % tail_feature;
		vec grad = sign(head_feature - tail_feature);

		head += -alpha * grad % relation_head
			+ alpha * relation_head % tail_feature / sum(feature) * sigma;
		relation_head += -alpha * grad % head
			+alpha * head % tail_feature / sum(feature) * sigma;
		tail += alpha * grad % relation_tail
			+alpha * relation_tail % head_feature / sum(feature) * sigma;
		relation_tail += alpha * grad % tail
			+alpha * tail % head_feature / sum(feature) * sigma;

		head = normalise(max(head, ones(dim) / pow(dim, 5)), 2);
		tail = normalise(max(tail, ones(dim) / pow(dim, 5)), 2);
		relation_head = normalise(max(relation_head, ones(dim) / pow(dim, 5)), 2);
		relation_tail = normalise(max(relation_tail, ones(dim) / pow(dim, 5)), 2);
	}

public:
	virtual void train_triplet(const pair<pair<int, int>, int>& triplet) override
	{
		pair<pair<int, int>, int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > margin)
			return;

		train_derv(triplet, alpha);
		train_derv(triplet_f, -alpha);
	}
};

class SFactorE
{
protected:
	vector<vec>	embedding_entity;
	vector<vec>	embedding_relation_head;
	vector<vec>	embedding_relation_tail;

public:
	const int		dim;
	const double	sigma;

public:
	int factor_index(const int entity_id) const
	{
		const vec& entity = embedding_entity[entity_id];

		unsigned int re_index;
		entity.max(re_index);

		return re_index;
	}

	SFactorE(int dim, int entity_count, int relation_count, double sigma)
		:dim(dim), sigma(sigma)
	{
		embedding_entity.resize(entity_count);
		for_each(embedding_entity.begin(), embedding_entity.end(),
			[=](vec& elem){elem = normalise(randu(dim), 2); });

		embedding_relation_head.resize(relation_count);
		for_each(embedding_relation_head.begin(), embedding_relation_head.end(),
			[=](vec& elem){elem = normalise(randu(dim), 2); });

		embedding_relation_tail.resize(relation_count);
		for_each(embedding_relation_tail.begin(), embedding_relation_tail.end(),
			[=](vec& elem){elem = normalise(randu(dim), 2); });
	}

	double prob(const pair<pair<int, int>, int>& triplet)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation_head = embedding_relation_head[triplet.second];
		vec& relation_tail = embedding_relation_tail[triplet.second];

		vec head_feature = head % relation_head;
		vec tail_feature = tail % relation_tail;

		return log(sum(head_feature % tail_feature)) * sigma
			- sum(abs(head_feature - tail_feature));
	}

	void train(const pair<pair<int, int>, int>& triplet, const double alpha)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation_head = embedding_relation_head[triplet.second];
		vec& relation_tail = embedding_relation_tail[triplet.second];

		vec head_feature = head % relation_head;
		vec tail_feature = tail % relation_tail;
		vec feature = head_feature % tail_feature;
		vec grad = sign(head_feature - tail_feature);

		head += -alpha * grad % relation_head
			+ alpha * relation_head % tail_feature / sum(feature) * sigma;
		relation_head += -alpha * grad % head
			+ alpha * head % tail_feature / sum(feature) * sigma;
		tail += alpha * grad % relation_tail
			+ alpha * relation_tail % head_feature / sum(feature) * sigma;
		relation_tail += alpha * grad % tail
			+ alpha * tail % head_feature / sum(feature) * sigma;

		head = normalise(max(head, ones(dim) / pow(dim, 5)), 2);
		tail = normalise(max(tail, ones(dim) / pow(dim, 5)), 2);
		relation_head = normalise(max(relation_head, ones(dim) / pow(dim, 5)), 2);
		relation_tail = normalise(max(relation_tail, ones(dim) / pow(dim, 5)), 2);
	}

public:
	void save(ofstream & fout)
	{
		storage_vmat<double>::save(embedding_entity, fout);
		storage_vmat<double>::save(embedding_relation_head, fout);
		storage_vmat<double>::save(embedding_relation_tail, fout);
	}

	void load(ifstream & fin)
	{
		storage_vmat<double>::load(embedding_entity, fin);
		storage_vmat<double>::load(embedding_relation_head, fin);
		storage_vmat<double>::load(embedding_relation_tail, fin);
	}

public:
	virtual vec entity_representation(int entity_id) const
	{
		return embedding_entity[entity_id];
	}
};

class MFactorE
	:public Model
{
protected:
	vector<SFactorE*>	factors;
	vector<vec>			relation_space;

protected:
	vector<vec>			acc_space;

public:
	const double	margin;
	const double	alpha;
	const int		dim;
	const int		n_factor;
	const double	sigma;

public:
	MFactorE(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		double sigma,
		int n_factor)
		:Model(dataset, task_type, logging_base_path),
		dim(dim), alpha(alpha), margin(training_threshold), n_factor(n_factor), sigma(sigma)
	{
		logging.record() << "\t[Name]\tMultiple.FactorE";
		logging.record() << "\t[Dimension]\t" << dim;
		logging.record() << "\t[Learning Rate]\t" << alpha;
		logging.record() << "\t[Training Threshold]\t" << training_threshold;
		logging.record() << "\t[Factor Number]\t" << n_factor;

		relation_space.resize(count_relation());
		for (vec& elem : relation_space)
		{
			elem = normalise(ones(n_factor));
		}

		for (auto i = 0; i < n_factor; ++i)
		{
			factors.push_back(new SFactorE(dim, count_entity(), count_relation(), sigma));
		}
	}

public:
	Col<int> factor_index(const int entity_id) const
	{
		Col<int> v_index(n_factor);
		for (auto i = 0; i < n_factor; ++i)
		{
			v_index[i] = factors[i]->factor_index(entity_id);
		}

		return v_index;
	}

	int category_index(const int entity_id, const int feature_id) const
	{
		return factors[feature_id]->factor_index(entity_id);
	}

	vec get_error_vec(const pair<pair<int, int>, int>& triplet) const
	{
		vec score(n_factor);
		auto i_score = score.begin();
		for (auto factor : factors)
		{
			*i_score++ = factor->prob(triplet);
		}

		return score;
	}

	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet) override
	{
		return sum(get_error_vec(triplet) % relation_space[triplet.second]);
	}

public:
	virtual void train_triplet(const pair<pair<int, int>, int>& triplet) override
	{
		pair<pair<int, int>, int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > margin)
			return;

		vec err = get_error_vec(triplet);
		vec err_f = get_error_vec(triplet);

		for (auto i=0; i<n_factor; ++i)
		{
			factors[i]->train(triplet, n_factor * alpha * relation_space[triplet.second][i]);
			factors[i]->train(triplet_f, - n_factor * alpha * relation_space[triplet.second][i]);
		}

		acc_space[triplet.second] += err;
	}

	virtual void train(bool last_time = false) override
	{
		acc_space.resize(count_relation());
		for (vec & elem : acc_space)
		{
			elem = zeros(n_factor);
		}

		Model::train(last_time);

		for (auto i = 0; i < count_relation(); ++i)
		{
			relation_space[i] = 
				normalise(max(-acc_space[i], ones(n_factor) / dim), 1);
		}
	}

public:
	virtual void save(const string& filename) override
	{
		ofstream fout(filename.c_str(), ios::binary);
		storage_vmat<double>::save(relation_space, fout);
		for (auto i = 0; i < n_factor; ++i)
		{
			factors[i]->save(fout);
		}
		fout.close();
	}

	virtual void load(const string& filename) override
	{
		ifstream fin(filename.c_str(), ios::binary);
		storage_vmat<double>::load(relation_space, fin);
		for (auto i = 0; i < n_factor; ++i)
		{
			factors[i]->load(fin);
		}
		fin.close();
	}

public:
	virtual vec entity_representation(int entity_id) const
	{
		vec rep_vec;
		for (auto i = 0; i < n_factor; ++i)
		{
			rep_vec = join_cols(rep_vec, factors[i]->entity_representation(entity_id));
		}
	
		return rep_vec;
	}
};

class MFactorSemantics
	:public MFactorE
{
protected:
	vector<vector<string>>		documents;
	map<string, vec>			topic_words;

public:
	vector<string>				tells;

public:
	MFactorSemantics(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		const string& semantic_file_raw,
		int dim,
		double alpha,
		double training_threshold,
		double sigma,
		int n_factor)
		:MFactorE(dataset, task_type, logging_base_path, dim, alpha, training_threshold,
		sigma, n_factor)
	{
		documents.resize(count_entity() + 10);
		tells.resize(count_entity() + 10);

		fstream fin(semantic_file_raw.c_str());
		boost::char_separator<char> sep(" \t \"\',.\\?!#%@");
		while (!fin.eof())
		{
			string strin;
			getline(fin, strin);
			transform(strin.begin(), strin.end(), strin.begin(), ::tolower);

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
						topic_words[*i] = zeros(dim * n_factor);
					}
				}
			}

			documents[data_model.entity_name_to_id.find(entity_name)->second] = entity_description;
			tells[data_model.entity_name_to_id.find(entity_name)->second] = strin;
		}
		fin.close();

		std::cout << "File Loaded." << endl;
	}

public:
	void analyze()
	{
		ofstream fout("D:\\Temp\\analyse.result");

		for (auto ient = 0; ient < count_entity(); ++ient)
		{
			vec rep_ent = entity_representation(ient);
			for (string& w : documents[ient])
			{
				topic_words[w] += rep_ent;
			}
		}

		for (auto & elem : topic_words)
		{
			for (auto i = 0; i < n_factor; ++i)
			{
				double t = 0;
				for (auto j = 0; j < dim; ++j)
				{
					t += elem.second[i*dim + j] * elem.second[i*dim + j];
				}

				t = sqrt(t);
				for (auto j = 0; j < dim; ++j)
				{
					elem.second[i*dim + j] /= t;
				}
			}
		}
	}

public:
	vector<int> infer_entity(string query, int n_item)
	{
		transform(query.begin(), query.end(), query.begin(), ::tolower);
		boost::char_separator<char> sep(" \t \"\',.\\?!#%@");
		boost::tokenizer<boost::char_separator<char>>	token(query, sep);

		vec rep_query = ones(dim * n_factor);
		for (auto i = token.begin(); i != token.end(); ++i)
		{
			rep_query = rep_query % topic_words[*i];
		}

		for (auto i = 0; i < n_factor; ++i)
		{
			double t = 0;
			for (auto j = 0; j < dim; ++j)
			{
				t += rep_query[i*dim + j] * rep_query[i*dim + j];
			}

			t = sqrt(t);
			for (auto j = 0; j < dim; ++j)
			{
				rep_query[i*dim + j] /= t;
			}
		}

		vector<pair<double, int>>	scores;
		for (auto i = 0; i < count_entity(); ++i)
		{
			scores.push_back(make_pair(dot(rep_query, entity_representation(i)), i));
		}

		sort(scores.begin(), scores.end(), greater<pair<double, int>>());
		vector<int> re;
		for (auto i = 0; i < n_item; ++i)
		{
			re.push_back(scores[i].second);
		}

		return re;
	}
};