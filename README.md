#Lua Neural Network Library

#Lua: https://lua.org

A simple neural network library for Lua with functions for activation functions and their derivatives,
forward and backward propagation as well as calculating cost functions like MSE or cross entropy.
Using an ID system for each neural network you can have multiple neural networks in 1 file, manage
train and visualize them easily and with no spaghetti code. (except this was made with spaghetti code,
use the spaghetti code to get rid of the spaghetti code.

    ::::DOCUMENTATION::::

the 'lnn.asserttype()' function has 3 parameters: variable (the variable you want to check the type of), variablename (the variable name for more verbose error messages) and thetype
(the type of variable you want to confirm it is). if the variable type is equal to the type of variable you want it to be it will continue the code, if not it will produce an error
message with information about the variable.

the 'lnn.assertsize()' function has 4 parameters: a (the first table), b (the second table), aname (the name of the first table, for more verbose error messages) and bname (the name
of the second table (for more verbose error messages). if the size of a is equal to the size of b it will continue the code, if not it will produce an error message with information
about the table sizes.

the 'lnn.findintable()' function has 2 parameters: item (the item you want to find in the table) and table (the table you want to find item in). it does what it sounds like, if it
finds item in table it will return true, if not it will return false.

the 'lnn.sigmoid()' function has 2 parameters: x (the number you want to put into the sigmoid function) and derivative (if you want the derivative of the sigmoid function). if the
derivative parameter is true it returns x put into the derivative of the sigmoid function and if it's false it returns x put into the sigmoid function.

the 'lnn.tanh()' function has 2 parameters: x (the number you want to put into the tanh function) and derivative (if you want the derivative of the tanh function). if the
derivative parameter is true it returns x put into the derivative of the tanh function and if it's false it returns x put into the tanh function.

the 'lnn.relu()' function has 2 parameters: x (the number you want to put into the relu function) and derivative (if you want the derivative of the relu function). if the
derivative parameter is true it returns x put into the derivative of the relu function and if it's false it returns x put into the relu function.

the 'lnn.leakyrelu()' function has 2 parameters: x (the number you want to put into the leakyrelu function) and derivative (if you want the derivative of the leakyrelu function).
if the derivative parameter is true it returns x put into the derivative of the leakyrelu function and if it's false it returns x put into the leakyrelu function.

the 'lnn.initialize()' function has 5 parameters: id (the id for the neural network), activation (the activation function for the neual network), insize (the size of the inputs for
the neural network) layercount (the amount of layers for the neural network) and outcount (the size of the output nodes for the neural network). it creates the data values for the
neural network, the weights, biases and current values. the amount of nodes in each layer is calculated from a 2 point slope, (y2-y1)/(x1-x2), where y2 is the output node count, y1
is the input node count, x1 is 0 and x2 is the layercount, the amount of hidden layers in the neural network.

the 'lnn.forwardpass()' function has 2 parameters: id (the id for the neural network) and intable (the input for the neural network). it propagates the input forward through the
neural network with the activation function as specified from the 'lnn.initialize()' function. it returns a table even if the amount of output nodes is 1. 
 
the 'lnn.adjust()' function has 5 parameters: id (the id for the neural network), intable (the input table to adjust the weights and biases with respect to), out (the real output
from the neural network), expectedout (the expected or ideal output from the neural network) and learningrate (usually a low value like 0.01 or 0.001). it adjusts the weights and
the biases by calculating gradw and gradb then subtracting gradw from the weights and subtracting gradb from the biases.

the 'lnn.getmse()' function has 2 parameters: output (the real output from the neural network) and expectedoutput (the expected or ideal output from the neural network). the functiom
returns the mean squared error calculated from the output and expected output.

the 'lnn.getsse()' function has 2 parameters: output (the real output from the neural network) and expectedoutput (the expected or ideal output from the neural network). the functiom
returns the sum of squared error calculated from the output and expected output.

the 'lnn.getrmse)' function has 2 parameters: output (the real output from the neural network) and expectedoutput (the expected or ideal output from the neural network). the functiom
returns the root of mean squared error calculated from the output and expected output.

the 'lnn.getcrossentropy()' function has 2 parameters: output (the real output from the neural network) and expectedoutput (the expected or ideal output from the neural network). the
function returns the cross entropy value calculated from the output and expected output. 

the 'lnn.debug.returnweights()' function has 1 parameter: id (the id for the neural network). the function returns a table with the weight values for the neural network. this can
be used for visualizing or debugging.

the 'lnn.debug.returnbiases()' function has 1 parameter: id (the id for the neural network). the function returns a table with the bias values for the neural network. this can be
used for visualizing or debugging.

the 'lnn.debug.returncurrent()' function has 1 parameter: id (the id for the neural network). the function returns a table with the current values (the node value of each layer) for
the neural network. this can be used for visualizing or debugging.

the 'lnn.debug.returnwgradient()' function has 1 parameter: id (the id for the neural network). the function returns a table with the gradient values for the neural network. this can
be used for visualizing or debugging. (or you can just use _G[id][gradient][gradw or gradb])

the 'lnn.debug.returndata()' function has 1 parameter: id (the id for the neural network). the function returns the table with the data for the neural network (info like layercount,
outcount, ect). (or you can just use _G[id])

    ::::HOW THE NEURAL NETWORK WORKS::::

the neural networks starts by creating the functions, the first and only function created is getlayer(). getlayer() takes the last layer, the next layer, the weight layer inbetween
and the bias table for the next layer. it then declares 1 variable, sum, and loops for the amount of items in the nextlayer (#nextlayer, loop variable is a). then in that loop there
is another loop that loops for the amount of items in the lastlayer (#lastlayer, loop variable is i), then it adds the i'th item in lastlayer multiplied by the i+(a-1) multiplied by
the amount of items in the last layer (#lastlayer). then the a'th item in the next layer (nextlayer) is set to sum + the current node's bias (biases[a]) put into the activation
function of choice.

the input is fed into the neural network in the form of a table, this is because it's easy to work with tables and if you want to read an image like BMP or PNG you would have to
write code to convert it to a table (or use https://github.com/Didericis/png-lua#pnglua :D) getlayer() is called for the input and the first hidden layer and then a loop for the
amount of hidden layers starting at 2 (loop variable is i) calls getlayer() for the last layer (_G[id.."c"..i-1]) and the next layer (_G[id.."c"..i]). once the loop is done it calles
getlayer() again for the lastlayer in the hidden layers and the output nodes. it then returns the output (_G[id.."o"]).

the adjust function starts by checking if out and expected out are the same size to prevent an error, it then gets the weighted sum of the weights and puts it into the derivative of
the activation function. before it does anything else it checks if the averaged mse for the output is below a certain value to avoid overfitting and if it is then return true. it then
calculates the gradient descent table for the weights and then the gradient descent table for the biases, it adjusts the output layer weights based on gradw and then the rest of the
weights, also based on gradw. after that it adjusts the output biases based on gradb, and then adjusts the rest of the biases, also based on gradb.

    ::::POTENTIAL OPTIMIZATIONS::::

some potential optimizations i copied from myself when i was at school during lunch at 11:05 am on thursday, febuary 16th, 2023 are are all for the back-propagation algorithm. there
are optimizations that are possible for the forward-pass algorithm but i'm not able to come up with any right now. after the output weights are adjusted it then adjusts the rest of
the weights, then again on the next gradw item, then again and so on. this can be optimized by subtracting all of them together and then subtracting the weights from that, same with
the biases.

another potential optimization is having a dataset parameter for the lnn.adjust() function. the dataset parameter could be a table with a list of inputs, outputs and expected outputs.

thats it i'm done with this it's 8:57 pm saturday febuary 18th 2023 and i've worked on this code and documentation for 10 hours straight i'll make it a luarocks package soon.
