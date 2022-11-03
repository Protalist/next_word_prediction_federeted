from globalVariable.global_variable import *
from globalVariable.model import *
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import sys


def server(num_client):
		NUM_CLIENTS=num_client
		KRUM=False


		class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
				def __init__(
						self,
						fraction_fit,
						fraction_eval,
						min_fit_clients,
						min_eval_clients,
						min_available_clients,
						initial_parameters,
						eval_fn_p
				):
						super(AggregateCustomMetricStrategy,self).__init__(
								fraction_fit=fraction_fit,  # Sample 10% of available clients for training
								fraction_eval=fraction_eval,  # Sample 5% of available clients for evaluation
								min_fit_clients=min_fit_clients,  # Never sample less than 10 clients for training
								min_eval_clients=min_eval_clients,  # Never sample less than 5 clients for evaluation
								min_available_clients=min_available_clients,  # Wait until at least 75 clients are available
								initial_parameters=initial_parameters
								)
						self.current_weigth = fl.common.parameters_to_weights(self.initial_parameters)
						self.eval_fn_p = eval_fn_p
						
				def aggregate_fit(
						self,
						rnd,
						results,
						failures,
				):
						if not results:
								return None, {}
						# Do not aggregate if there are failures and failures are not accepted
						if not self.accept_failures and failures:
								return None, {}

						# Convert results
						weights_results = [
								(fl.common.parameters_to_weights(fit_res.parameters), fit_res.num_examples,fit_res.metrics['cid'])
								for _, fit_res in results
						]            
						self.Weight_update_statistics(weights_results,rnd)
						weights_results = self.accuracy_checking(weights_results,rnd)
						if KRUM:
								weights_results = self.krum(weights_results,rnd)
								self.current_weigth=self.aggregate(weights_results)
						else:
							self.current_weigth=self.aggregate(weights_results)
						parameters_aggregated = fl.common.weights_to_parameters(self.current_weigth)

						# Aggregate custom metrics if aggregation fn was provided
						metrics_aggregated = {}
						return parameters_aggregated, metrics_aggregated

				def krum(self, result, rnd):
						score = {}
						for r in result:
							distances=[]
							for d in result:
								if d[2] == r[2]:
									continue
								d = distance_lp_norm(d[0],r[0])
								distances.append(d)
							distances.sort()
							score[r[2]] = distances[2:]
						
						delta_krum = min(score, key=score.get)
						return [item for item in result if item[2] == delta_krum]

				def accuracy_checking(self,results,rnd):
						ret = []
						for r in results:
								print(f"check client {r[2]}")
								W_i = self.aggregate([r])
								W_g_i = self.aggregate([x for x in results if x[2] != r[2]])
								_, dict_w_i = self.eval_fn_p(W_i)
								_, dict_g_i = self.eval_fn_p(W_g_i)
								trashold=-0.02
								if rnd > 7:
									trashold=-0.01
								if dict_w_i["val_top_3"]-dict_g_i["val_top_3"] >= trashold:
										round = pickle.load(open(acuracy_checking_path, 'rb'))
										round[str(rnd)+"-"+str(r[2])]={"agent no malicius": r[2],"distance": dict_w_i["val_top_3"]-dict_g_i["val_top_3"] }
										pickle.dump(round, open(acuracy_checking_path, 'wb'))
										ret.append(r)
								else:
										print(r[2], "is malicius")
										round = pickle.load(open(acuracy_checking_path, 'rb'))
										round[str(rnd)+"-"+str(r[2])]={"agent malicius": r[2],"distance": dict_w_i["val_top_3"]-dict_g_i["val_top_3"] }
										pickle.dump(round, open(acuracy_checking_path, 'wb'))

						return ret

				def aggregate(self,results) :
						"""Compute weighted average."""
						# Calculate the total number of examples used during training
						num_examples_total = sum([num_examples for _, num_examples, _ in results])

						# Create a list of weights, each multiplied by the related number of examples
						weighted_weights = [
								[layer * num_examples for layer in weights] for weights, num_examples, _ in results
						]

						# Compute average weights of each layer
						weights_prime = [
								reduce(np.add, layer_updates) / num_examples_total
								for layer_updates in zip(*weighted_weights)
						]
						return np.array(self.current_weigth, dtype=object)+np.array(weights_prime,dtype=object)

				def aggregate_evaluate(
						self,
						rnd: int,
						results,
						failures,
				) :
						#Aggregate evaluation losses using weighted average
						if not results:
										return None
						
						loss,_ = super().aggregate_evaluate(rnd, results, failures)

						# Weigh accuracy of each client by number of examples used
						accuracies = [r.metrics["val_accuracy"] * r.num_examples for _, r in results]
						top_k = [r.metrics["val_top_3"] * r.num_examples for _, r in results]
						examples = [r.num_examples for _, r in results]

						# Aggregate and print custom metric
						accuracy_aggregated = sum(accuracies) / sum(examples)
						top_k_grragated = sum(top_k) / sum(examples)
						print(f"Round {rnd}/15 aggregated  from client results loss : {loss},accuracy : {accuracy_aggregated} , top_3: {top_k_grragated} ")

						# Call aggregate_evaluate from base class (FedAvg)
						return (loss, {"accuracy":accuracy_aggregated,"top_k":top_k_grragated})

				def Weight_update_statistics(self, results, rnd: int):
					if len(results)<=1:
						return

					agents = results.copy()
					results = {}
					ret = []
					for m in agents:
						R_max =float('-inf')
						R_min =float('inf')

						R_max_m = float('-inf')
						R_min_m =float('inf')

						for a in agents:
							if a[2] == m[2]:
								continue
							d = distance_weigths_scalar(m[0],a[0])
							if R_max_m < d:
								R_max_m = d
							if R_min_m > d :
								R_min_m = d

						for a in agents:
							if a[2] == m[2]:
								continue
							for a2 in agents:
								if a[2]==a2[2] or a2[2]==m[2]:
									continue
								d = distance_weigths_scalar(a2[0],a[0])
								if R_max < d:
									R_max = d
								if R_min > d:
									R_min = d

						r=max(abs(R_max_m-R_min),abs(R_min_m-R_max))
						results[m[2]] = r
					
					for r in results:
						avarege = 0
						for r2 in results:
							if r == r2:
								continue
							avarege = avarege + results[r2]
						k=2.5
						diff = abs(results[r]-(avarege/(len(results)-1)))
						if(diff>k):
							print(f"agent malicius {r} detected wiyh distance {results[r]}")
							round = pickle.load(open(weigth_update_statistics_path, 'rb'))
							round[str(rnd)+"-"+str(r)]={"agent": r,"distance":results[r]}
							pickle.dump(round, open(weigth_update_statistics_path, 'wb'))
						else:
							print(f"agent {r} is not malicius with distance {results[r]}")
					return


		def get_eval_fn(model):
				"""Return an evaluation function for server-side evaluation."""

				def get_data():            
						sequences = pickle.load(open(r'globalVariable\sequences.pk1', 'rb'))
						import random
						random.shuffle(sequences)
						partition_size = math.floor(len(sequences) / (NUM_CLIENTS))
						idx_from, idx_to =  (1+1) * partition_size, (1+2) * partition_size

						X_f = []
						y_f = []

						for i in sequences[idx_from: idx_to]:
								X_f.append(i[0:lengt_sequence])
								y_f.append(i[-1])
								
						X_f = np.array(X_f)
						y_f = np.array(y_f)
						return X_f,y_f

				# The `evaluate` function will be called after every round
				def evaluate(weights: fl.common.Weights):
						model.set_weights(weights)  # Update model with the latest parameters
						x_val, y_val = get_data()
						loss, accuracy,top_3 = model.evaluate(x_val, y_val)
						return loss, { "loss":loss,"val_accuracy":accuracy , "val_top_3": top_3}


				return evaluate


		vocab_dict = pickle.load(open(r'globalVariable\token.pk1', 'rb'))
		vocab_size = len(vocab_dict)
		
		model = next_word_model(vocab_size,lengt_sequence)
		pickle.dump({}, open(weigth_update_statistics_path, 'wb'))
		pickle.dump({}, open(acuracy_checking_path, 'wb'))

		strategy=AggregateCustomMetricStrategy(
						fraction_fit=0.4,  # Sample 10% of available clients for training
						fraction_eval=0.2,  # Sample 5% of available clients for evaluation
						min_fit_clients=3,  # Never sample less than 10 clients for training
						min_eval_clients=2,  # Never sample less than 5 clients for evaluation
						min_available_clients=int(NUM_CLIENTS * 0.75),  # Wait until at least 75 clients are available
						initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
						#fit_metrics_aggregation_fn =fit_metrics_aggregation_fn_custom
						eval_fn_p=get_eval_fn(model)
		)

		hist=fl.server.start_server(server_address ="localhost:3031",config={"num_rounds": 15}, strategy=strategy)

		pickle.dump(hist, open(r'model\result_poisoned.pk1', 'wb'))

if __name__ == "__main__":
		print (sys.argv)
		b=0
		try:
				b = int(sys.argv[1])
		except IndexError:
				b = 10

		server(b)



		#https://www.googleapis.com/geolocation/v1/geolocate?key=%GOOGLE_LOCATION_SERVICE_API_KEY%