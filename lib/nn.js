// Other techniques for learning

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function dsigmoid(y) {
  // return sigmoid(x) * (1 - sigmoid(x));
  return y * (1 - y);
}


class NeuralNetwork {
  constructor(input_nodes, hidden_layers, output_nodes) {

    // making the old notation work
    if(!hidden_layers instanceof Array){
      let tmp = [hidden_layers];
      hidden_layers =tmp;
    }

    this.input_nodes = input_nodes;
    this.output_nodes = output_nodes;
    //this is an array
    this.hidden_layers = hidden_layers;

    //put all layers in one array
    this.layers = [input_nodes].concat(this.hidden_layers).concat([this.output_nodes]);

    //initializing the weights of all layers
    this.weights = new Array(this.layers.length-1);
    for(let i=0;i<this.weights.length;i++){
      this.weights[i]=new Matrix(this.layers[i+1],this.layers[i]);
      this.weights[i].randomize();
    }

    //initializing the biases
    this.biases = new Array(this.layers.length-1);
    for(let i=0;i<this.biases.length;i++){
      this.biases[i]=new Matrix(this.layers[i+1],1);
      this.biases[i].randomize();
    }

    this.setLearningRate(0.05);
  }

  predict(input_array) {

    // Generating the Hidden Outputs
    let inputs = Matrix.fromArray(input_array);

    let prev = inputs;
    for(let i=0;i<this.weights.length;i++){
      let x = Matrix.multiply(this.weights[i],prev);  // let hidden = Matrix.multiply(this.weights_ih, inputs);
      x.add(this.biases[i]);  // hidden.add(this.bias_h);
      //activation function!
      x.map(sigmoid); // hidden.map(sigmoid);
      prev=x;
    }

    let output = prev;

    // Sending back to the caller!
    return output.toArray();
  }

  setLearningRate(learning_rate = 0.1) {
    this.learning_rate = learning_rate;
  }

  train(input_array, target_array) {
    console.log(this.weights);
    let input = Matrix.fromArray(input_array);
    let prev = input;
    let hidden_values = []; //called hidden prevously
    for(let i=0;i<this.weights.length;i++){
      let x = Matrix.multiply(this.weights[i],prev);  // let hidden = Matrix.multiply(this.weights_ih, inputs);
      x.add(this.biases[i]);  // hidden.add(this.bias_h);
      //activation function!
      x.map(sigmoid); // hidden.map(sigmoid);
      hidden_values.push(x);
      prev=x;
    }
    let outputs = prev;
    console.log(this.weights);
    let weights_t =[];
    for(let weight of this.weights){
      weights_t.push(Matrix.transpose(weight));
    }
    console.log(this.weights);
    // Convert array to matrix object
    let targets = Matrix.fromArray(target_array);

    // Calculate the error
    // ERROR = TARGETS - OUTPUTS
    let output_errors = Matrix.subtract(targets, hidden_values[hidden_values.length-1]);

    let errors = new Array(weights_t.length);
    errors[weights_t.length-1]=output_errors;
    //console.log("Error array",errors);
    //console.log("transposed weights arr",weights_t);
    for (let i=weights_t.length-2;i>0;i--) {
      //console.log("Err i+1",errors[i+1]);
      //console.log("Weights_t i",weights_t[i+1])
      errors[i]=Matrix.multiply(weights_t[i+1],errors[i+1]);
    }

    let gradients = [];
    for(let i=0;i<hidden_values.length;i++){
      let g = Matrix.map(hidden_values[i],dsigmoid);
      g.multiply(errors[i]);
      g.multiply(this.learning_rate);
      gradients.push(g);
    }

    let deltas = [];
    for(let i=0;i<hidden_values.length;i++){
      let val = Matrix.transpose(hidden_values[i]);
      deltas.push(Matrix.multiply(gradients[i],val));
    }

    for(let i=0;i<this.weights.length;i++){
      this.weights[i].add(deltas[i]);
      this.biases[i].add(gradients[i]);
    }

    /*

    // let gradient = outputs * (1 - outputs);
    // Calculate gradient
    let gradients = Matrix.map(hidden_values[hidden_values.length-1], dsigmoid);
    gradients.multiply(output_errors);
    gradients.multiply(this.learning_rate);


    // Calculate deltas
    let hidden_T = Matrix.transpose(hidden);
    let weight_ho_deltas = Matrix.multiply(gradients, hidden_T);

    // Adjust the weights by deltas
    this.weights_ho.add(weight_ho_deltas);
    // Adjust the bias by its deltas (which is just the gradients)
    this.bias_o.add(gradients);

    // Calculate the hidden layer errors
    let who_t = Matrix.transpose(this.weights_ho);
    let hidden_errors = Matrix.multiply(who_t, output_errors);

    // Calculate hidden gradient
    let hidden_gradient = Matrix.map(hidden, dsigmoid);
    hidden_gradient.multiply(hidden_errors);
    hidden_gradient.multiply(this.learning_rate);

    // Calcuate input->hidden deltas
    let inputs_T = Matrix.transpose(inputs);
    let weight_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T);

    this.weights_ih.add(weight_ih_deltas);
    // Adjust the bias by its deltas (which is just the gradients)
    this.bias_h.add(hidden_gradient);

    // outputs.print();
    // targets.print();
    // error.print();
    */
  }

}
