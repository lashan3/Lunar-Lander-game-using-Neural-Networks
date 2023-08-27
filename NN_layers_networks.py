import NN_hyperparams
import NN_neuron
import math
from csv import reader

from NN_hyperparams import training_flag, iterations,nn_lambda, nn_lr, nn_momentum , hidden_neurons, output_neurons, input_neurons,x1_max,x2_max,y1_max,y2_max,x1_min,x2_min,y1_min,y2_min
from NN_neuron import neuron_struct
#Defining Neural Network layers
#input layer
input = []
#hidden layer
hidden = []
#output layer
output = []
breakpoint = 0.0
global de_normalize
de_normalize = False

#Initializing layers with neurons
for i in range(0 , input_neurons+1):
  input.append(neuron_struct(i,0,2))

for i in range(0 , hidden_neurons+1):
  hidden.append(neuron_struct(i,0,2))
  
for i in range(0 , output_neurons):
  output.append(neuron_struct(i,0,0))

#function to normalize the game input row
def normalization(inputs):
  input_one = (inputs[0] - (x1_min)) / ((x1_max) - (x1_min)) 
  input_two = (inputs[1] - (x2_min)) / ((x2_max) - (x2_min))
  normalized_result = [input_one, input_two]
  return normalized_result

#function to denormalize the precited output
def de_normalization(outputs):
  output_one = (outputs[0] * ((y1_max) - (y1_min)) + (y1_min))
  output_two = (outputs[1] * ((y2_max) - (y2_min)) + (y2_min))
  denormalized_result = [abs(output_one) if (de_normalize) else -abs(output_two), abs(output_two)]
  return denormalized_result

#Error calculation on training file
def error_epoch():
  error_epoch = []
  with open('normalized_training.csv', 'r') as training:
    data = reader(training)
    for row in data:
      feed_forward([row[0], row[1], 1]) 
      error_epoch.append( ( (float(row[2]) - float(output[0].activation))**2 + (float(row[3]) - float(output[1].activation))**2 ) / 2)               
    return math.sqrt(sum(error_epoch) / len(error_epoch))

#Error calculation on validation file
def error_validate(): 
  total_error = []
  with open('normalized_testing.csv', 'r') as validating:
    data = reader(validating)
    for row in data:
      feed_forward([row[0], row[1], 1])
      total_error.append( ( (float(row[2]) - float(output[0].activation))**2 + (float(row[3]) - float(output[1].activation))**2 ) / 2)                                
    return math.sqrt(sum(total_error) / len(total_error)) 

def feed_forward(input_row):
  #Input layer activations
  for neuron in range(0, len(input)):
      input[neuron].activation = input_row[neuron]
  #hidden layer activations
  for neuron in range(0, (len(hidden) - 1)):
      hidden[neuron].weigths_calculation(input)
  #Setting for bias
  hidden[-1].activation = 1
  #output layer activations
  for neuron in range(0, len(output)):
      output[neuron].weigths_calculation(hidden)
  return

def back_propogation(outputs_network):
  error = []
  
  #calculating error
  for neuron in range(0, len(output)):
      error.append(float(outputs_network[neuron]) - float(output[neuron].activation))
      
  #gradient for output layer    
  for neuron in range(0, len(output)):
      output[neuron].gradiant = nn_lambda * output[neuron].activation * (1 - output[neuron].activation) * error[neuron] 

  #gradient for hidden layer where we dont have error
  for neuron in range(0, len(hidden)):
      result = 0
      for i in range(0, len(output)):
          result = result + ( float(output[i].gradiant) * float(hidden[neuron].weights[i]) ) 
      
      hidden[neuron].gradiant = nn_lambda * hidden[neuron].activation * (1 - hidden[neuron].activation) * result
      
  #calculating delta weight for hidden layer
  for neuron in range(0, len(hidden)):
      for i in range(0, len(output)): 
          hidden[neuron].d_weights[i] = nn_lr * float(output[i].gradiant) * float(hidden[neuron].activation) + nn_momentum * float(hidden[neuron].d_weights[i])
  
  #calculating delta weights for input layer
  for neuron in range(0, len(input)):
      for i in range(0, (len(hidden) - 1)): 
          input[neuron].d_weights[i] = nn_lr * float(hidden[i].gradiant) * float(input[neuron].activation) + nn_momentum * float(input[neuron].d_weights[i])
  
  #updating weights
  for neuron in range(0, len(hidden)):
      for i in range(0, len(hidden[neuron].weights)):
          hidden[neuron].weights[i] = float(hidden[neuron].weights[i]) + float(hidden[neuron].d_weights[i])

  for neuron in range(0, len(input)):
      for i in range(0, len(input[neuron].weights)):
          input[neuron].weights[i] = float(input[neuron].weights[i]) + float(input[neuron].d_weights[i])
  return

#Training Function
def training():
  with open('normalized_training.csv', 'r') as training:
    data = reader(training)
    for row in data:
      #feed forward against each input row in data
      feed_forward([row[0], row[1], 1])
      #feed backward against each input row in data
      back_propogation([row[2], row[3]])
  
#Function to save weights in txt file
def weight_save():
  file = open("neural_network_weights.txt", "w")
  for i in range(0, len(input)):
    for j in range(0, len(input[i].weights)):
      file.write(str(input[i].weights[j]) + ",")
      
  for i in range(0, len(hidden)):
    for j in range(0, len(hidden[i].weights)):
      file.write(str(hidden[i].weights[j]) + ",")        
  file.close()
  return

#fetch weights from txt file and initialize weights into layers
def weight_fetch_and_initialize(row):
  weights = []
  with open('neural_network_weights.txt') as f:
    line = f.read()
    weights = line.split(",")     
  flag = 0
  for i in range(0, len(input)):
    for j in range(0, len(input[i].weights)):
      input[i].weights[j] = weights[flag]
      flag = flag + 1
  global de_normalize
  de_normalize = True if(row[0] > 0) else False
  for i in range(0, len(hidden)):
    for j in range(0, len(hidden[i].weights)):
      hidden[i].weights[j] = weights[flag]
      flag = flag + 1
  return

#Function to do prediction against game input
def predictions(row):
  #initialize weight
  weight_fetch_and_initialize(row)
  #normalize game row
  normalized_result = normalization(row)
  #add bias in input row
  normalized_result.append(1)
  #feed forward on normalized input row
  feed_forward(normalized_result)
  #denormalizing output and return it to game
  denormalized_result = de_normalization([output[0].activation,output[1].activation])
  return denormalized_result


print("-----------------------------------")
print('Input layer', len(input))
print('Output layer', len(output))
print('Hidden layer', len(hidden))
print("-----------------------------------")
if training_flag == 1:
  #Running epochs for training
  for i in range(iterations):
    training()  
    training_error = error_epoch()
    #Stopping condition for training
    if float("{:.7f}".format(training_error)) == float("{:.7f}".format(breakpoint)):
        print("Training stopped due to defined stopping criteria")
        break
    else:
       breakpoint = training_error   
    validation_error = error_validate() 
    print("-----------------------------------")
    print('Epoch : ', i + 1, ' Training Error : ', training_error, ' Validation Error : ',  validation_error)
    i += 1
  #Saving weights after training is finished
  weight_save()
  print('Training Ended')
else:
  #doing random prediction
  predictions([556.311,354.100])