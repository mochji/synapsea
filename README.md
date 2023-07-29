# Synapsea
### (formerly Lua Neural Network)

## Lua: https://lua.org

## NOTE

This is kinda bad right now, uhh yeah. I don't know when v1.3.00 is gonna come out but just letting you know, this is kinda bad right now !

## DESCRIPTION

A simple neural network library for Lua with functions for activation functions and their derivatives,
forward and backward propagation as well as calculating cost functions like MSE or cross entropy.
Using an ID system for each neural network you can have multiple neural networks in 1 file, manage
train and visualize them easily and with no spaghetti code. (except this was made with spaghetti code,
use the spaghetti code to get rid of the spaghetti code.)

## CONTACT INFO

Youtube: <a href="https://youtube.com/@xxoa_/">@xxoa_</a>

Github: <a href="https://github.com/x-xxoa">x-xxoa</a>

tumblr: <a href="https://x-xxoa.tumblr.com">@x-xxoa</a>

Email: xxoa.yt@gmail.com

## VERSION NUMBERING SCHEME

x.y.zz-abc

x is the major version, y is the minor version, zz is the patch version and abc is the branch like `fast` or `unstable`.

A patch is usually a small change to a function or 2, a minor version is usually a large change to a lot of functions or a system overhaul aswell as changes to documentation and the major version goes up on the first release of the year that makes the minor version number go up. If the minor version reaches a number above 9 in 1 year somehow it will just go to 10 and so on.

Example:

`v1.2.00-unstable` would mean the unstable version of the 2nd minor version of the 1st major version.

`v1.1.02` would mean the stable version of the 2nd patch of the 1st minor version of the 1st major version.

`v2.3.08-fast` would mean the fast version of the 8th patch of the 3rd minor version of the 2nd major version.

## DEFAULT ACTIVATION & ERROR FUNCTIONS

### DEFAULT ACTIVATION FUNCTIONS
 - Sigmoid ("sigmoid")
 - Hyperbolic Tangent ("tanh")
 - ReLU ("relu")
 - LReLU ("leakyrelu")
 - ELU ("elu")
 - Swish ("swish")
 - Binary Step ("binarystep")
 - Softmax **NOTE:** Cannot be used as the activation function in the default neural network functions. (well you can but it will error because x is a number.)
 - Softplus ("softplus")
 - Softsign ("softsign")
 - Linear ("linear")

### DEFAULT COST/ERROR FUNCTIONS
 - MSE (Mean Squared Error)
 - MAE (Mean Absolute Error)
 - SSE (Sum of Squared Error)
 - RMSE (Root of Mean Squared Error)
 - Cross Entropy
 - Categorical Cross Entropy

## STRENGTHS AND LIMITATIONS

### STRENGTHS

 - Ability to run in LuaJIT.

 - No external libraries & can run off bare Lua (5.1.x and above).

 - ID system for managing neural networks with clean code.

 - Support for different activations on different layers.

 - Simple and easy to understand structure.

### LIMITATIONS

 - _G[] can be slower and clutter variables

 - There might not be as many features of neural network libraries in other programming languages or even other ones in Lua. the main point of this is that it's easy and simple with no spaghetti code.

 - Slightly slower due to error checking, check the 'fast' branch for no error checking. **NOTE:** the fast branch might not be updated at the same speed that the 'stable' branch is.

 - The `lnn.adjust()` function doesn't support custom loss functions at the moment. (but you could create your own back-propagation algorithm for the neural networks since they're easy to interface with !)

 - No support for CNNs. (yet)

## ALL FUNCTIONS

### ERROR CATCHING/USEFUL FUNCTIONS

 - [**lnn.asserttype()**](#at)
 - [**lnn.assertsize()**](#as)
 - [**lnn.findintable()**](#fit)
 - [**lnn.sumtable()**](#st)

### DEFAULT ACTIVATION FUNCTIONS

 - [**lnn.activation.sigmoid()**](#sig)
 - [**lnn.activation.tanh()**](#ta)
 - [**lnn.activation.relu()**](#re)
 - [**lnn.activation.leakyrelu()**](#lre)
 - [**lnn.activation.elu()**](#el)
 - [**lnn.activation.swish()**](#sw)
 - [**lnn.activation.binarystep()**](#bs)
 - [**lnn.activation.softmax()**](#sm)
 - [**lnn.activation.softplus()**](#sp)
 - [**lnn.activation.softsign()**](#ss)
 - [**lnn.activation.linear()**](#li)

### NEURAL NETWORK FUNCTIONS

 - [**lnn.initialize()**](#in)
 - [**lnn.forwardpass()**](#fp)
 - [**lnn.returnerror()**](#rte)
 - [**lnn.adjust.adjustfromgradient()**](#afg)
 - [**lnn.adjust.basic.adjust()**](#ad)
 - [**lnn.adjust.basic.returngradient()**](#rg)
 - [**lnn.adjust.momentum.adjust()**](#mad)
 - [**lnn.adjust.momentum.returngradient()**](#mrg)

### DEFAULT LOSS FUNCTIONS

 - [**lnn.loss.mse()**](#mse)
 - [**lnn.loss.mae()**](#mae)
 - [**lnn.loss.sse()**](#sse)
 - [**lnn.loss.rmse()**](#rmse)
 - [**lnn.loss.crossentropy()**](#ce)
 - [**lnn.loss.categoricalcrossentropy()**](#cce)

### DATA FUNCTIONS

 - [**lnn.data.randomize()**](#ra)
 - [**lnn.data.addrandom()**](#ar)
 - [**lnn.data.exportdata()**](#ed)

# NEURAL NETWORK TABLE STRUCTURES

## NEURAL NETWORK TABLE

```
id = {                     --TYPES:        DESCRIPTION:
	activations            --table         table containing activations for each layer
	alpha                  --number        used as a multiplication constant in some activation functions, 1 by default, 0.01 if activation is leakyrelu
	gradient               --table         (see the section below)
	id                     --string        id for the neural network
	weight                 --table         table containing tables containing the weights, numbered by the next layer so weight 1 is connected to node 1 in the 2nd layer and node 1 in the 1st layer and weight 2 is connected to node 1 in the 2nd layer and node 2 in the 1st layer for example
	bias                   --table         table containing the bias for each layer, every hidden layer has a bias and is added to the layer after it.
	current                --table         table containing tables containing the values of the node
	layersizes             --table         table containing the sizes of the input, hidden and output layers
	weightcount            --number        the total amount of weights in the neural network
}
```

## GRADIENT TABLE

```
grad = {                   --TYPES:        DESCRIPTION:
	error,                 --table         error of every node
	grad = {               --table         table containing the gradients for the weights and biases
		weight             --table         table containing the gradient for every weight
		bias               --table         table containing the gradient for every bias
	},
	learningrate,          --number        step size for training, usually a low number like 0.01 or 0.001
	momentum               --number        momentum, not used in this function.
}
```

# OTHER STUFF

Have questions? Email me or add a question on the issues page with the tag `question`. Find a bug? Describe it and tag it with `bug`, `minor bug`, `major bug` or `edge case` on the issues page. Anything else? If applicable, add it on the issues page with the correct tag, I'll get to it... hopefully.

# FUNCTION DOCUMENTATION

## ERROR CATCHING/USEFUL FUNCTIONS

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="at"></a>
   # lnn.asserttype()
   ### DESCRIPTION
   Compares the type of `variable` against `expectedtype`.
   
   ### PARAMETERS
   | parameter name   | type       | description                                               |
   |------------------|------------|-----------------------------------------------------------|
   | **variable**     | **any**    | The variable that you want to check the type of.          |
   | **variablename** | **string** | The name of the variable for more verbose error messages. |
   | **expectedtype** | **string** | The expected type of the variable.                        |
   
   This function allows you to check the type of a variable to prevent or catch errors before they happen. This function compares `type(variable)` to `expectedtype` and calls `error()` with a string formatted like so if the 2 strings are different:

   >"`variablename` (`variable`) is not a `expectedtype` or is nil. Type: `type(variable)`"
   
</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="as"></a>
   # lnn.assertsize()
   ### DESCRIPTION
   Compares the size of `a` against `b`.
   
   ### PARAMETERS
   | parameter name | type        | description                                            |
   |----------------|-------------|--------------------------------------------------------|
   | **a**          | **table**   | The first table.                                       |
   | **b**          | **table**   | The second table.                                      |
   | **aname**      | **string**  | The name of table `a` for more verbose error messages. |
   | **bname**      | **string**  | The name of table `b` for more verbose error messages. |
   | **zerocheck**  | **boolean** | Should the function error if either size is 0.         |
   
   This function allows you to check the size of 2 tables to prevent or catch errors before they happen. This function compares `#a` to `#b` and calls `error()` with a string formatted like so if the 2 numbers are different:

   >"`aname` (`#a`) is not the same size as `bname` (`#b`)."

   Additionally, if `zerocheck` is true, if either one of the table sizes is 0 it will call `error()` with a string formatted like this:

   >"`aname` (`#a`) or `bname` (`#b`) is equal to zero."

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="fit"></a>
   # lnn.findintable()
   ### DESCRIPTION
   Performs a linear search on `table` to find `item`.
   
   ### PARAMETERS
   | parameter name | type        | description                                                                   |
   |----------------|-------------|-------------------------------------------------------------------------------|
   | **table**      | **table**   | The table you want to find `item` in.                                         |
   | **item**       | **any**     | The item you want to find in `table`.                                         |
   | **recursive**  | **boolean** | Should the function call itself when the current index of `table` is a table. |

   This function performs a linear search on `table` to find `item`, if `recursive` is true and the current index of `table` is a table it will call the lnn.findintable() function on the current table index. If it finds `item` in `table` it will return the current table index number (the i in the for loop, i'm not good with words) and if it doesn't find `item` it will return false. 


</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="st"></a>
   # lnn.sumtable()
   ### DESCRIPTION
   Adds all items in `table` to get the sum of the table.
   
   ### PARAMETERS
   | parameter name | type       | description                                      |
   |----------------|------------|--------------------------------------------------|
   | **table**      | **table**  | The table you want to get the sum of. |
   
   This function allows you to get the sum of all numbers in a table.

</div>

## DEFAULT ACTIVATION FUNCTIONS

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="sig"></a>
   # lnn.activation.sigmoid()
   ### DESCRIPTION
   Returns `x` put into the sigmoid function if `derivative` is false and returns `x` put into the derivative of the sigmoid function if `derivative` is true.
   
   ### PARAMETERS
   | parameter name   | type        | description                                                                   |
   |------------------|-------------|-------------------------------------------------------------------------------|
   | **x**            | **number**  | The number you want to put into the sigmoid function.                         |
   | **derivative**   | **boolean** | Should the function return x put into the derivative of the sigmoid function. |
   
   If `derivative` is false this function returns `x` put into the sigmoid activation function, otherwise if `derivative` is true it returns `x` put into the derivative of the sigmoid activation function.

   ### MATHEMATICAL FUNCTION
   $$\frac{1}{1+e^{-x}}$$

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="ta"></a>
   # lnn.activation.tanh()
   ### DESCRIPTION
   Returns `x` put into the Tanh (Hyperbolic Tangent) function if `derivative` is false and returns `x` put into the derivative of the Tanh function if `derivative` is true.
   
   ### PARAMETERS
   | parameter name   | type        | description                                                                   |
   |------------------|-------------|-------------------------------------------------------------------------------|
   | **x**            | **number**  | The number you want to put into the tanh function.                            |
   | **derivative**   | **boolean** | Should the function return x put into the derivative of the tanh function.    |
   
   If `derivative` is false this function returns `x` put into the Tanh activation function, otherwise if `derivative` is true it returns `x` put into the derivative of the Tanh activation function.

   ### MATHEMATICAL FUNCTION
   $$\frac{e^{2x}-1}{e^{2x}+1}$$

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="re"></a>
   # lnn.activation.relu()
   ### DESCRIPTION
   Returns `x` put into the ReLU (Rectified Linear Unit) function if `derivative` is false and returns `x` put into the derivative of the ReLU function if `derivative` is true.
   
   ### PARAMETERS
   | parameter name | type        | description                                                                |
   |----------------|-------------|----------------------------------------------------------------------------|
   | **x**          | **number**  | The number you want to put into the ReLU function.                         |
   | **derivative** | **boolean** | Should the function return x put into the derivative of the ReLU function. |
   
   If `derivative` is false this function returns `x` put into the ReLU activation function, otherwise if `derivative` is true it returns `x` put into the derivative of the ReLU activation function.

   ### MATHEMATICAL FUNCTION
   $$max(0,x)$$

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="lre"></a>
   # lnn.activation.leakyrelu()
   ### DESCRIPTION
   Returns `x` put into the LeakyReLU (Leaky Rectified Linear Unit) activation function if `derivative` is false and retuns `x` put into the derivative of the LReLU activation function if `derivative` is true.
   
   ### PARAMETERS
   | parameter name | type        | description                                                                 |
   |----------------|-------------|-----------------------------------------------------------------------------|
   | **x**          | **number**  | The number you want to put into the LReLU activation function.              |
   | **derivative** | **boolean** | Should the function return x put into the derivative of the LReLU function. |
   
   If `derivative` is false this function returns `x` put into the LReLU activation function, otherwise if `derivative` is true, it returns `x` put into the derivative of the LReLU activation function.

   ### MATHEMATICAL FUNCTION
   $$
   \begin{cases}
	   x,&\text{if } x\gt0\\
       x\cdot\alpha,&\text{otherwise}
   \end{cases}
   $$

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="el"></a>
   # lnn.activation.elu()
   ### DESCRIPTION
   Returns `x` put into the ELU (Exponential Linear Unit) activation function if `derivative` is false and retuns `x` put into the derivative of the ELU activation function if `derivative` is true.
   
   ### PARAMETERS
   | parameter name | type        | description                                                                                                       |
   |----------------|-------------|-------------------------------------------------------------------------------------------------------------------|
   | **x**          | **number**  | The number you want to put into the ELU activation function.                                                      |
   | **derivative** | **boolean** | Should the function return x put into the derivative of the ELU function.                                         |
   | **alpha**      | **number**  | Kind of like... scale? of the graph? This is why you should do research too! I'm not good at explaining in words! |
   
   If `derivative` is false this function returns `x` put into the ELU activation function, otherwise if `derivative` is true, it returns `x` put into the derivative of the ELU activation function.

   ### MATHEMATICAL FUNCTION
   $$
   \begin{cases}
	   x,&\text{if } x\gt0\\
	   e^{x}-1,&\text{otherwise}
   \end{cases}
   $$

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="sw"></a>
   # lnn.activation.swish()
   ### DESCRIPTION
   Returns `x` put into the Swish activation function if `derivative` is false and retuns `x` put into the derivative of the Swish activation function if `derivative` is true.
   
   ### PARAMETERS
   | parameter name | type        | description                                                                 |
   |----------------|-------------|-----------------------------------------------------------------------------|
   | **x**          | **number**  | The number you want to put into the Swish activation function.              |
   | **derivative** | **boolean** | Should the function return x put into the derivative of the Swish function. |
   | **alpha**      | **number**  | Again, just kind of like... just a.. constant that is multiplied somewhere. |
   
   If `derivative` is false this function returns `x` put into the Swish activation function, otherwise if `derivative` is true, it returns `x` put into the derivative of the Swish activation function.

   ### MATHEMATICAL FUNCTION
   $$
   \frac{x}{1+e^{(-\alpha)\cdot\text{x}}}
   $$

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="bs"></a>
   # lnn.activation.binarystep()
   ### DESCRIPTION
   Returns 1 if `x` > 0, returns 0 if `x` =< 0.
   
   ### PARAMETERS
   | parameter name | type        | description                                                                       |
   |----------------|-------------|-----------------------------------------------------------------------------------|
   | **x**          | **number**  | The number you want to put into the binary step activation function.              |
   | **derivative** | **boolean** | Should the function return x put into the derivative of the binary step function. |
   
   If `derivative` is false this function returns 1 if `x` > 0 and returns 0 if `x` =< 0 otherwise if `derivative` is true, it returns 0.

   ### MATHEMATICAL FUNCTION
   $$
   \begin{cases}
   1,\text{if }x > 0\\
   0,\text{otherwise }
   \end{cases}
   $$

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="sm"></a>
   # lnn.activation.softmax()
   ### DESCRIPTION
   Returns `x` put into the Softmax activation function if `derivative` is false and retuns `x` put into the derivative of the Softmax activation function if `derivative` is true.

   ### NOTE:
   `x` for this function is a table, not a number!
   
   ### PARAMETERS
   | parameter name | type        | description                                                                              |
   |----------------|-------------|------------------------------------------------------------------------------------------|
   | **x**          | **table**   | The table you want to put into the Softmax activation function.                          |
   | **derivative** | **boolean** | Should the function return x put into the derivative of the Softmax activation function. |

   If `derivative` is false this function returns `x` put into the Softmax activation function, otherwise if `derivative` is true, it returns `x` put into the derivative of the Softmax activation function.

   ### MATHEMATICAL FUNCTION
   $$
   \sum_{i=1}^{|A|x}\frac{{e^{x_{i}}}}{\sum_{a=1}^{|A|x}e^{x_{a}}}
   $$

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="sp"></a>
   # lnn.activation.softplus()
   ### DESCRIPTION
   Returns `x` put into the Softplus activation function if `derivative` is false and retuns `x` put into the derivative of the Softplus activation function if `derivative` is true.
   
   ### PARAMETERS
   | parameter name | type        | description                                                                               |
   |----------------|-------------|-------------------------------------------------------------------------------------------|
   | **x**          | **number**  | The number you want to put into the Softplus activation function.                         |
   | **derivative** | **boolean** | Should the function return x put into the derivative of the Softplus activation function. |

   If `derivative` is false this function returns `x` put into the Softplus activation function, otherwise if `derivative` is true, it returns `x` put into the derivative of the Softplus activation function.

   ### MATHEMATICAL FUNCTION
   $$
   log(1+e^{x})
   $$

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="ss"></a>
   # lnn.activation.softsign()
   ### DESCRIPTION
   Returns `x` put into the Softsign activation function if `derivative` is false and retuns `x` put into the derivative of the Softsign activation function if `derivative` is true.
   
   ### PARAMETERS
   | parameter name | type        | description                                                                               |
   |----------------|-------------|-------------------------------------------------------------------------------------------|
   | **x**          | **number**  | The number you want to put into the Softsign activation function.                         |
   | **derivative** | **boolean** | Should the function return x put into the derivative of the Softsign activation function. |

   If `derivative` is false this function returns `x` put into the Softsign activation function, otherwise if `derivative` is true, it returns `x` put into the derivative of the Softsign activation function.

   ### MATHEMATICAL FUNCTION
   $$
   \frac{x}{1+|x|}
   $$

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="li"></a>
   # lnn.activation.linear()
   ### DESCRIPTION
   Returns `x` if `derivative` is false and retuns 1 if `derivative` is true. This is mainly just for cleaner code so there's not a special statment for the `linear` activation function.
   
   ### PARAMETERS
   | parameter name | type        | description                    |
   |----------------|-------------|--------------------------------|
   | **x**          | **number**  | The number you want to return. |
   | **derivative** | **boolean** | Should the function return 1.  |
   | **alpha**      | **number**  | Number to be multiplied by x.  |

   ### MATHEMATICAL FUNCTION
   $$
   x
   $$

</div>

## NEURAL NETWORK FUNCTIONS

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="in"></a>
   # lnn.initialize()
   ### DESCRIPTION
   Creates a neural network with an id of `id` and creates all the required data for it in a table in _G.

   ### PARAMETERS
   | parameter name | type       | description                                                                       |
   |----------------|------------|-----------------------------------------------------------------------------------|
   | **id**         | **string** | The id for the neural network, cannot be the same as another neural network's id. |
   | **activation** | **string** | Should be the name of a function in `lnn.activation`.                             |
   | **layersizes** | **table**  | The table containing the sizes for each layer in the neural network.              |

   This function creates a neural network with an id of `id` and creates all the data for it. the `activation` parameter should be the name of a function in `lnn.activation`, so "sigmoid" would be a valid activation function and if you made your own function in `lnn.activation` that would also be a valid activation function. The `layersizes` parameter should be a table containing the sizes for all layers. The first number in `layersizes` would be the input size, the last number in `layersizes` would be the output size and the numbers inbetween would be the hidden layer sizes. You can create a neural network with no hidden layers by just having 2 numbers, input and output size.

   ### RETURN VALUE
   Nothing

   ### EXAMPLE

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="fp"></a>
   # lnn.activation.forwardpass()
   ### DESCRIPTION
   Forward propagates the intable through the neural network with an id of `id`. (gets the output of `id` with input `intable`)

   ### PARAMETERS
   | parameter name | type       | description                                                 |
   |----------------|------------|-------------------------------------------------------------|
   | **id**         | **string** | The id of the neural network you want to get the output of. |
   | **intable**    | **table**  | Input for the neural network.                               |

   Gets the output of `id` with input `intable`, go to the how the neural network works section for information on how this function works.

   ### RETURN VALUE
   Neural network output (`_G[id]["current"][_G[id]["layercount"]+1]`)

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="rte"></a>
   # lnn.returnerror()
   ### DESCRIPTION
   Calculates the error of neural network `id` with input `intable`, output `output` and expected/ideal output `expectedoutput`.

   ### PARAMETERS
   | parameter name     | type       | description                                                                                  |
   |--------------------|------------|----------------------------------------------------------------------------------------------|
   | **id**             | **string** | The id of the neural network you want to adjust.                                             |
   | **intable**        | **table**  | Input for the neural network.                                                                |
   | **output**         | **table**  | The real output for the neural network.                                                      |
   | **expectedoutput** | **table**  | The expected/ideal output for the neural network with an input of `intable`.                 |
   | **learningrate**   | **number** | THe step size for adjusting the weights and biases, usually a low number like 0.01 or 0.001. |

   ### RETURN VALUE
   Error of nodes with respect to the parameters.

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="afg"></a>
   # lnn.adjust.adjustfromgradient()
   ### DESCRIPTION
   Adjusts `id`'s weights and biases from `gradient`.

   ### PARAMETERS
   | parameter name | type       | description                                                                                                          |
   |----------------|------------|----------------------------------------------------------------------------------------------------------------------|
   | **id**         | **string** | The id of the neural network you want to adjust.                                                                     |
   | **gradient**   | **table**  | Table containing the  data, assumed to be returned from `lnn.adjust.**.returngradient()` or with the same structure. |

   ### RETURN VALUE
   Nothing

</div>

## BASIC ADJUST FUNCTIONS

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="ad"></a>
   # lnn.adjust.basic.adjust()
   ### DESCRIPTION
   Calculates the gradient of neural network `id` with input `intable`, output `output` and expected/ideal output `expectedoutput` and adjusts `id`'s weights and biases for stochastic gradient descent without momentum.

   ### PARAMETERS
   | parameter name     | type       | description                                                                                  |
   |--------------------|------------|----------------------------------------------------------------------------------------------|
   | **id**             | **string** | The id of the neural network you want to adjust.                                             |
   | **intable**        | **table**  | Input for the neural network.                                                                |
   | **output**         | **table**  | The real output for the neural network.                                                      |
   | **expectedoutput** | **table**  | The expected/ideal output for the neural network with an input of `intable`.                 |
   | **learningrate**   | **number** | THe step size for adjusting the weights and biases, usually a low number like 0.01 or 0.001. |

   ### RETURN VALUE
   Nothing

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="rg"></a>
   # lnn.adjust.basic.returngradient()
   ### DESCRIPTION
   Calculates the gradient of neural network `id` with input `intable`, output `output` and expected/ideal ouitput of `expectedoutput and returns the gradient and error without momentum. Could be used for batch or mini-batch gradient descent.

   ### PARAMETERS
   | parameter name     | type       | description                                                                                  |
   |--------------------|------------|----------------------------------------------------------------------------------------------|
   | **id**             | **string** | The id of the neural network you want to calculate the gradient of.                          |
   | **intable**        | **table**  | Input for the neural network.                                                                |
   | **output**         | **table**  | The real output for the neural network.                                                      |
   | **expectedoutput** | **table**  | The expected/ideal output for the neural network with an input of `intable`.                 |
   | **learningrate**   | **number** | THe step size for adjusting the weights and biases, usually a low number like 0.01 or 0.001. |

   ### RETURN VALUE
   Gradient

</div>

## MOMENTUM ADJUST FUNCTIONS

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="mad"></a>
   # lnn.adjust.momentum.adjust()
   ### DESCRIPTION
   Calculates the gradient of neural network `id` with input `intable`, output `output` and expected/ideal output `expectedoutput` and adjusts `id`'s weights and biases for stochastic gradient descent with momentum.

   ### PARAMETERS
   | parameter name     | type       | description                                                                                  |
   |--------------------|------------|----------------------------------------------------------------------------------------------|
   | **id**             | **string** | The id of the neural network you want to adjust.                                             |
   | **intable**        | **table**  | Input for the neural network.                                                                |
   | **output**         | **table**  | The real output for the neural network.                                                      |
   | **expectedoutput** | **table**  | The expected/ideal output for the neural network with an input of `intable`.                 |
   | **learningrate**   | **number** | THe step size for adjusting the weights and biases, usually a low number like 0.01 or 0.001. |

   ### RETURN VALUE
   Nothing

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="mrg"></a>
   # lnn.adjust.momentum.returngradient()
   ### DESCRIPTION
   Calculates the gradient of neural network `id` with input `intable`, output `output` and expected/ideal ouitput of `expectedoutput and returns the gradient and error with momentum. Could be used for batch or mini-batch gradient descent.

   ### PARAMETERS
   | parameter name     | type       | description                                                                                  |
   |--------------------|------------|----------------------------------------------------------------------------------------------|
   | **id**             | **string** | The id of the neural network you want to calculate the gradient of.                          |
   | **intable**        | **table**  | Input for the neural network.                                                                |
   | **output**         | **table**  | The real output for the neural network.                                                      |
   | **expectedoutput** | **table**  | The expected/ideal output for the neural network with an input of `intable`.                 |
   | **learningrate**   | **number** | THe step size for adjusting the weights and biases, usually a low number like 0.01 or 0.001. |

   ### RETURN VALUE
   Gradient

</div>

## DEFAULT LOSS FUNCTIONS

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="mse"></a>
   # lnn.loss.mse()
   ### DESCRIPTION
   Returns the Mean Squared Error of `output` minus `expectedoutput`.

   ### PARAMETERS
   | parameter name     | type      | description                |
   |--------------------|-----------|----------------------------|
   | **output**         | **table** | The real output.           |
   | **expectedoutput** | **table** | The expected/ideal output. |

   ### MATHEMATICAL FUNCTION
   $$(\frac{\sum_{i=1}^{|A|o}o-eo}{|A|o})^{2}$$

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="mae"></a>
   # lnn.loss.mae()
   ### DESCRIPTION
   Returns the Mean Absolute Error of `output` minus `expectedoutput`.

   ### PARAMETERS
   | parameter name     | type      | description                |
   |--------------------|-----------|----------------------------|
   | **output**         | **table** | The real output.           |
   | **expectedoutput** | **table** | The expected/ideal output. |

   ### MATHEMATICAL FUNCTION
   $$|\frac{\sum_{i=1}^{|A|o}o-eo}{|A|o}|$$

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="sse"></a>
   # lnn.loss.sse()
   ### DESCRIPTION
   Returns the Sum of Squared Error of `output` minus `expectedoutput`.

   ### PARAMETERS
   | parameter name     | type      | description                |
   |--------------------|-----------|----------------------------|
   | **output**         | **table** | The real output.           |
   | **expectedoutput** | **table** | The expected/ideal output. |

   ### MATHEMATICAL FUNCTION
   $$\sum_{i=1}^{|A|o}o-eo^{2}$$

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="rmse"></a>
   # lnn.loss.rmse()
   ### DESCRIPTION
   Returns the Root of Mean Squared Error of `output` minus `expectedoutput`.

   ### PARAMETERS
   | parameter name     | type      | description                |
   |--------------------|-----------|----------------------------|
   | **output**         | **table** | The real output.           |
   | **expectedoutput** | **table** | The expected/ideal output. |

   ### MATHEMATICAL FUNCTION
   $$\sqrt((\frac{\sum_{i=0}^{|A|o}o-eo}{|A|o})^{2})$$

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="ce"></a>
   # lnn.loss.crossentropy()
   ### DESCRIPTION
   Returns the Cross Entropy of `output` and `expectedoutput`.

   ### PARAMETERS
   | parameter name     | type      | description                |
   |--------------------|-----------|----------------------------|
   | **output**         | **table** | The real output.           |
   | **expectedoutput** | **table** | The expected/ideal output. |

   ### MATHEMATICAL FUNCTION
   $$\sum_{i=1}^{|A|o}{eo_{i}\cdot\log(o_{i})}$$

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="cce"></a>
   # lnn.loss.categoricalcrossentropy()
   ### DESCRIPTION
   Returns the Categorical Cross Entropy of `output` and `expectedoutput`.

   ### PARAMETERS
   | parameter name     | type      | description                |
   |--------------------|-----------|----------------------------|
   | **output**         | **table** | The real output.           |
   | **expectedoutput** | **table** | The expected/ideal output. |

   ### MATHEMATICAL FUNCTION
   $$\sum_{i=1}^{|A|o}{eo_{i}\cdot log(o_{i})}$$

</div>

## DATA FUNCTIONS

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="ra"></a>
   # lnn.data.randomize()
   ### DESCRIPTION
   Randomizes `id`'s weights and biases.

   ### PARAMETERS
   | parameter name | type       | description                                                  |
   |----------------|------------|--------------------------------------------------------------|
   | **id**         | **string** | The id that you want to randomize the weights and biases of. |

   This function sets `id`'s weights and biases to a random number between -1 and 1.

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="ar"></a>
   # lnn.data.addrandom()
   ### DESCRIPTION
   Adds a random amount between `upperlimit` and `lowerlimit` to `id`'s weights and biases.

   ### PARAMETERS
   | parameter name | type       | description                                                               |
   |----------------|------------|---------------------------------------------------------------------------|
   | **id**         | **string** | The id that you want to add a random amount to the weights and biases of. |
   | **lowerlimit** | **number** | The lower limit.                                                          |
   | **upperlimit** | **number** | The upper limit.                                                          |

   This function adds a random amount between `upperlimit` and `lowerlimit` to `id`'s weights and biases, could be useful when training a stickman to walk or something like that.

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="ed"></a>
   # lnn.data.exportdata()
   ### DESCRIPTION
   Exports `id`'s data to `filename`

   ### PARAMETERS
   | parameter name | type       | description                                                 |
   |----------------|------------|-------------------------------------------------------------|
   | **id**         | **string** | The id that you want to export.                             |
   | **filename**   | **string** | The name of the file you want the id's data to be saved to. |

   This function exports the data in `_G[id]` to `filename`.

</div>

You reached the bottom, if you see this, hello?
