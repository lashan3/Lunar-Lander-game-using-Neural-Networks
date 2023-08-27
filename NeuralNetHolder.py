from NN_layers_networks import predictions
class NeuralNetHolder:

    def __init__(self):
        super().__init__()

    
    def predict(self, input_row):
        input = input_row.split(",") 
        output = predictions([float(input[0]), float(input[1])])
        return output
    
    
    