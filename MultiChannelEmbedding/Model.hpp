#pragma once
#include "Import.hpp"
#include "ModelConfig.hpp"
#include "DataModel.hpp"

using namespace std;
using namespace arma;

class Model
{
protected:
	const DataModel		data_model;
	const TaskType		task_type;

public:
	ModelLogging		logging;

public:
	int	epos;

public:
	Model(	const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path)
		:data_model(dataset), task_type(task_type), logging(logging_base_path)
	{
		epos = 0;
		best_triplet_result = 0;
		std::cout<<"Ready"<<endl;

		logging.record()<<"\t[Dataset]\t"<<dataset.name;
		logging.record()<<TaskTypeName(task_type);
	}

public:
	virtual double prob_triplets(const pair<pair<int, int>,int>& triplet) = 0;
	virtual void train_triplet(const pair<pair<int, int>,int>& triplet) = 0;

public:
	virtual void train(bool last_time = false)
	{
		++ epos;

#pragma omp parallel for
		for(auto i=data_model.data_train.begin(); i!=data_model.data_train.end(); ++i)
		{
			train_triplet(*i);
		}
	}

	void run(int total_epos)
	{
		logging.record()<<"\t[Epos]\t"<<total_epos;

		-- total_epos;
		while(total_epos --> 0)
		{
			std::cout<<epos<<',';
			train();

			if (task_type == TripletClassification)
				test_triplet_classification();
		}

		train(true);
	}

public:
	double		best_triplet_result;
	double		best_link_mean;
	double		best_link_hitatten;
	double		best_link_fmean;
	double		best_link_fhitatten;

	void reset()
	{
		best_triplet_result = 0;
		best_link_mean = 1e10;
		best_link_hitatten = 0;
		best_link_fmean = 1e10;
		best_link_fhitatten = 0;
	}

	void test(int hit_rank = 10)
	{
		logging.record();
		
		best_link_mean = 1e10;
		best_link_hitatten = 0;
		best_link_fmean = 1e10;
		best_link_fhitatten = 0;

		if (task_type == LinkPredictionHead || task_type == LinkPredictionTail || task_type == LinkPredictionRelation)
			test_link_prediction(hit_rank);
		else
			test_triplet_classification();
	}

public:
	void test_triplet_classification()
	{
		double real_hit = 0;
		for(auto r=0; r<data_model.set_relation.size(); ++r)
		{
			vector<pair<double, bool>>	threshold_dev;
			for(auto i=data_model.data_dev_true.begin(); i!=data_model.data_dev_true.end(); ++i)
			{
				if (i->second != r)
					continue;

				threshold_dev.push_back(make_pair(prob_triplets(*i), true));
			}
			for(auto i=data_model.data_dev_false.begin(); i!=data_model.data_dev_false.end(); ++i)
			{
				if (i->second != r)
					continue;

				threshold_dev.push_back(make_pair(prob_triplets(*i), false));
			}

			sort(threshold_dev.begin(), threshold_dev.end());

			double threshold;
			double vari_mark = 0;
			int total = 0;
			int hit = 0;
			for(auto i=threshold_dev.begin(); i!=threshold_dev.end(); ++i)
			{
				if (i->second == false)
					++ hit;
				++ total;

				if (vari_mark <= 2*hit - total + data_model.data_dev_true.size())
				{
					vari_mark = 2*hit - total + data_model.data_dev_true.size();
					threshold = i->first;
				}
			}

			double lreal_hit = 0;
			double lreal_total = 0;
			for(auto i=data_model.data_test_true.begin(); i!=data_model.data_test_true.end(); ++i)
			{
				if (i->second != r)
					continue;

				++ lreal_total;
				if (prob_triplets(*i) > threshold)
					++ real_hit, ++lreal_hit;
			}

			for(auto i=data_model.data_test_false.begin(); i!=data_model.data_test_false.end(); ++i)
			{
				if (i->second != r)
					continue;

				++ lreal_total;
				if (prob_triplets(*i) <= threshold)
					++ real_hit, ++ lreal_hit;
			}

			//logging.record()<<data_model.relation_id_to_name.at(r)<<"\t"
			//	<<lreal_hit/lreal_total;
		}

		std::cout<<epos<<"\t Accuracy = "
			<<real_hit/(data_model.data_test_true.size() + data_model.data_test_false.size());
		best_triplet_result = max(
			best_triplet_result, 
			real_hit/(data_model.data_test_true.size() + data_model.data_test_false.size()));
		std::cout<<", Best = "<<best_triplet_result<<endl;

		logging.record()<<epos<<"\t Accuracy = "
			<<real_hit/(data_model.data_test_true.size() + data_model.data_test_false.size())
			<<", Best = "<<best_triplet_result;
	}

	void test_link_prediction(int hit_rank = 10, const bool is_relation = false)
	{
		double mean = 0;
		double hits = 0;
		double fmean = 0;
		double fhits = 0;
		double total = data_model.data_test_true.size();

		double arr_mean[20] = {0};
		double arr_total[5] = {0};

		for(auto i=data_model.data_test_true.begin(); i!=data_model.data_test_true.end(); ++i)
		{
			++ arr_total[data_model.relation_type[i->second]];
		}

		int cnt = 0;

#pragma omp parallel for
		for(auto i=data_model.data_test_true.begin(); i!=data_model.data_test_true.end(); ++i)
		{
			++cnt;
			if (cnt%100 == 0)
			{
				std::cout<<cnt<<',';
			}

			pair<pair<int, int>, int> t = *i;
			int frmean = 0;
			int rmean = 0;
			double score_i = prob_triplets(*i);

			if (task_type == LinkPredictionRelation || is_relation == true)
			{
				for(auto j=0; j!=data_model.set_relation.size(); ++j)
				{
					t.second = j;

					if (score_i >= prob_triplets(t))
						continue;

					++ rmean;

					if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
						++ frmean;
				}
			}
			else
			{
				for(auto j=0; j!=data_model.set_entity.size(); ++j)
				{
					if (task_type == LinkPredictionHead)
						t.first.first = j;
					else
						t.first.second = j;

					if (score_i >= prob_triplets(t))
						continue;

					++ rmean;

					if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
					{
						++ frmean;
					}
				}
			}

#pragma omp critical
			{
				if (frmean < hit_rank)
					++ arr_mean[data_model.relation_type[i->second]];

				mean += rmean;
				fmean += frmean;
				if (rmean < hit_rank)
					++ hits;
				if (frmean< hit_rank)
					++ fhits;
			}
		}

		std::cout<<endl;
		for(auto i=1; i<=4; ++i)
		{
			std::cout<<i<<':'<<arr_mean[i]/arr_total[i]<<endl;
			logging.record()<<i<<':'<<arr_mean[i]/arr_total[i];
		}
		logging.record();

		best_link_mean = min(best_link_mean, mean/total);
		best_link_hitatten = max(best_link_hitatten, hits/total);
		best_link_fmean = min(best_link_fmean, fmean/total);
		best_link_fhitatten = max(best_link_fhitatten, fhits/total);

		std::cout<<"Raw.BestMEANS = "<<best_link_mean<<endl;
		std::cout<<"Raw.BestHITS = "<<best_link_hitatten<<endl;
		logging.record()<<"Raw.BestMEANS = "<<best_link_mean;
		logging.record()<<"Raw.BestHITS = "<<best_link_hitatten;
		std::cout<<"Filter.BestMEANS = "<<best_link_fmean<<endl;
		std::cout<<"Filter.BestHITS = "<<best_link_fhitatten<<endl;
		logging.record()<<"Filter.BestMEANS = "<<best_link_fmean;
		logging.record()<<"Filter.BestHITS = "<<best_link_fhitatten;
	}

	virtual void draw(const string& filename, const int radius, const int id_relation) const
	{
		return;
	}

	virtual void draw(const string& filename, const int radius, 
		const int id_head, const int id_relation)
	{
		return;
	}

	virtual void report(const string& filename) const
	{
		return;
	}
public:
	~Model()
	{
		logging.record();
	}

public:
	int count_entity() const
	{
		return data_model.set_entity.size();
	}

	int count_relation() const
	{
		return data_model.set_relation.size();
	}
	
	const DataModel& get_data_model() const
	{
		return data_model;
	}
};