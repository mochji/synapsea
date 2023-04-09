# Lua Neural Network Library

## Lua: https://lua.org

Description:
A simple neural network library for Lua with functions for activation functions and their derivatives,
forward and backward propagation as well as calculating cost functions like MSE or cross entropy.
Using an ID system for each neural network you can have multiple neural networks in 1 file, manage
train and visualize them easily and with no spaghetti code. (except this was made with spaghetti code,
use the spaghetti code to get rid of the spaghetti code.)

## CONTACT INFO

Youtube: <a href="https://youtube.com/@xxoa_/">@xxoa_</a>

Github: <a href="https://github.com/x-xxoa">x-xxoa</a>

Email: xxoa.yt@gmail.com

## VERSION FORMATTING

x.y.zz-abc

x is the major version, y is the minor version, zz is the patch version and abc declares if it's unstable or not.

A patch is usually a small change to a function or 2, a minor version is usually a large change to a lot of functions or a system overhaul aswell as changes to documentation and the major version goes up when the minor version reaches a number bigger than 9.

Example:

`v1.2.00-unstable` would mean the unstable version of the 3rd patch of the 2nd minor version of the 1st major version.

`v1.1.02` would mean the stable version of the 2nd patch of the 1st minor version of the 1st major version.

## DEFAULT ACTIVATION & ERROR FUNCTIONS

### DEFAULT ACTIVATION FUNCTIONS
 - Sigmoid ("sigmoid")
 - Hyperbolic Tangent ("tanh")
 - ReLU ("relu")
 - LReLU ("leakyrelu")
 - ELU ("elu")
 - Swish ("swish")
 - Binary Step ("binarystep")
 - Linear ("linear")
 - Softmax **NOTE:** All layers when using `lnn.initialize()` must be equal when using this as `activation`.

### DEFAULT COST/ERROR FUNCTIONS
 - MSE (Mean Squared Error)
 - MAE (Mean Absolute Error)
 - SSE (Sum of Squared Error)
 - RMSE (Root of Mean Squared Error)
 - Cross Entropy
 - Binary Cross Entropy
 - Categorical Cross Entropy

## STRENGTHS AND LIMITATIONS

### STRENGTHS

 - Because this library uses Lua it has the ability to be run in LuaJIT which is really fast.

 - Because this uses no external libraries that means that all this requires is bare Lua 5.4 to run.

 - Because it uses an id system and integrates them into the functions it's easy to manage the neural networks with clean and readable code.

 - Because there are many default activation functions and support for custom activation functions it can fit a lot of use cases. ( not all but a lot :) )

 - Because the structure of the neural networks is simple and made easy to understand, you could create your own functions involving the neural networks if the built-in default ones do not fit your needs.

### LIMITATIONS

 - Because it uses _G[] to store the neural network values it count be slow and clutter variables.

 - Because this is in Lua there might not be as many features of neural network libraries in other programming languages or even other ones in Lua. the main point of this is that it's easy and simple with no spaghetti code. (for you, there's metric tons of spaghetti code in the functions)

 - Because of the error checking, this isn't as fast as it could be. check the 'fast' branch for no error checking. **NOTE:** the fast branch might not be updated at the same speed that the 'stable' branch is.

 - The `lnn.adjust()` function doesn't support custom loss functions at the moment.

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

### NEURAL NETWORK FUNCTIONS

 - [**lnn.initialize()**](#in)
 - [**lnn.forwardpass()**](#fp)
 - [**lnn.adjust()**](#ad)
 - [**lnn.returngradient()**](#rg)
 - [**lnn.adjustfromgradient()**](#afg)

### DEFAULT LOSS FUNCTIONS

 - [**lnn.loss.mse()**](#mse)
 - [**lnn.loss.mae()**](#mae)
 - [**lnn.loss.sse()**](#sse)
 - [**lnn.loss.rmse()**](#rmse)
 - [**lnn.loss.crossentropy()**](#ce)
 - [**lnn.loss.binarycrossentropy()**](#bce)
 - [**lnn.loss.categoricalcrossentropy()**](#cce)

### DEBUG FUNCTIONS

 - [**lnn.debug.returnweights()**](#rw)
 - [**lnn.debug.returnbiases()**](#rb)
 - [**lnn.debug.returncurrent()**](#rc)
 - [**lnn.debug.clearid()**](#ci)
 - [**lnn.debug.randomize()**](#ra)

### DATA FUNCTIONS

 - [**lnn.data.exportdata()**](#ed)
 - [**lnn.data.importdata()**](#id)

# HOW THE NEURAL NETWORK WORKS

## CREATING THE NEURAL NETWORK
The `lnn.initialize()` function creates a table with this structure in _G: (check out the documentation on `lnn.initialize()` if you dont know the parameters for the function)
```
_G[id] = {                 --TYPES:        ASSIGNED: (on lnn.initialize())                          DESCRIPTION:
	activation,            --string        activation                                               activation function for neural network, in lnn.activation
	layercount,            --number        #layersizes-2                                            amount of hidden layers
	outcount,              --number        #layersizes[#layersizes]                                 output size
	insize,                --number        #layersizes[1]                                           input size
	alpha,                 --number        1 by default, 0.01 if activation is "leakyrelu"          multiplication constant for some activation functions
	gradient = {           --table                                                                  gradient (actually calculated in lnn.adjust() or lnn.returngradient())
		gradw,             --table         empty                                                    weight gradient for output nodes
		gradb,             --table         empty                                                    bias gradient for output nodes
		gwsum,             --number        0                                                        sum of weight gradient for hidden layer nodes
		gbsum,             --number        0                                                        sum of bias gradient for hidden layer nodes
		dwinsum,           --number        0                                                        derivative of the activation function of the weighted sum of the input
		learningrate       --number        0                                                        step size, usually a low number like 0.01 or 0.001
	},                             
	id,                    --string        id                                                       id for the neural network
	weight,                --table         (covered later)                                          table containing tables containing weights
	bias,                  --table         (covered later)                                          table containing tables containing biases
	current,               --table         (covered later)                                          table containing tables containing node values
	layersizes             --table         layersizes                                               sizes of every layer
}
```

Then for `#layersizes-1`, it creates these tables: (a is for loop variable)
```
_G[id]["current"][a] = {}
_G[id]["bias"][a] = {}
_G[id]["weight"][a] = {}
```
The reason it is `#layersizes-1` is because we need the last layer (a) and the next (a+1) in some cases like creating the weight tables.

Then for `layersizes[a+1]` (i is for loop variable) it fills `_G[id]["current"][a][i]` with 0s and fills `_G[id]["bias"][a][i]` with a random number between -1 and 1. The reason it uses `layersizes[a+1]` is because `layersizes[1]` is the size for the input, we're creating the hidden layer (or output layer if `#layersizes` is 2, so no hidden layers) bias and current tables.

Then the `amounttofill` variable is set to `layersizes[a+1]*layersizes[a]` and for `amounttofill` (i is again for loop variable) it sets `_G["weight"][a][i]` to a random number between -1 and 1.

## GETTING THE OUTPUT

Now that we've created our neural network, we need to get the output. To calculate a node we need to multiply each node in the last layer by it's weight that is connected to the current node in the layer we're currently on, sum it together, put that in the activation function and then repeat this for every node in the next layer to get the rest of the layer. So if we had 4 input nodes and 2 output nodes, this is what it would look like to get `o1` and `o2`:

```
o1 = act( (i1*w1) + (i2*w2) + (i3*w3) + (i4*w4) )
o2 = act( (i1*w5) + (i2*w6) + (i3*w5) + (i4*w6) )
```

**NOTE:** `act()` is the activation function, there are many but the default ones are listed above and you can create your own in `lnn.activation`.

Notice how on `o2` we continued from `w5`? That's because it's a different weight and in `lnn.forwardpass()`, it assumes that the weights is next layer based. (i just made that up, idk what else to call it)

So if we had a more complex neural network, we would multiply each node in the last layer by it's weight that is connected to the current node in the layer we're currently on, sum it together, put that into the activation function, repeat that for every node in the next layer and then repeat that for every layer in the neural network.

## TRAINING

**NOTE:** I tried to actually create a back-propagation algorithm but it failed miserably but the default one included works well enough to use. If you're looking for a tutorial on proper back-propagation please research it yourself. This is just describing how to default one included works.

It starts by getting the weighted sum of the input, once that is calculated it calculates `gradw` and `gradb`.

Weight gradient: (`gradw`)
For every output node it gets the difference between (`output[i]` and `expectedoutput[i]` squared times `dwinsum` (the thing we calculated earlier) times `learningrate`) plus (the difference between `output[i]` and `expectedoutput[i]` times `learningrate`)

Bias gradient: (`gradb`)
For every output node it gets the difference between (`output[i]` and `expectedoutput[i]` times `learningrate`) plus (the difference between `output[i]` and `expectedoutput[i]` times `learningrate`)

The reason we get the difference between `output[i]` and `expectedoutput[i]` twice in both is to prevent the vanishing gradient problem in a very simple way.

We then calculate `gwsum` by getting the sum of `gradw` and we calculate `gbsum` by getting the sum of `gradb`. The `gradient` table on `_G[id]` is then updated.

The output node weights are adjusted with `gradw` and the output node biases are adjusted wtih `gradb`. The hidden layer weights are then adjusted with `gwsum` and the hidden layer biases are adjusted with `gbsum`.

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
   
   ### EXAMPLE
```lua
function examplefunction(x)
	--check if x is a number
	lnn.asserttype(x,"x","number")
	--[[
		if x is a table like this:
		{1,2}
		it will call error() with this message:
		"x (table: 0xMemoryaddress) is not a number or is nil. Type: table"
	]]--
end
```

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

   ### EXAMPLE
```lua
function exampleloss(real,ideal)
	--check if real and ideal are tables
	lnn.asserttype(real,"real","table")
	lnn.asserttype(idea,"ideal","table")
	
	--check sizes
	lnn.assertsize(real,idea,"real","ideal",true)
end
```

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

   ### EXAMPLE
```lua
local exampletable = {
	"this",
	"is",
	"an",
	"example",
	"i'm",
	"bad",
	"with",
	"examples"
}
if lnn.findintable(exampletable,user_input,false) then
	print("found input in exampletable")
else
	print("could not find input in exampletable")
end
```

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
   
   ### EXAMPLE
```lua
function get_layers_sum(id)
	local t = lnn.debug.returncurrent(id)
	local return_t = {}
	for i = 1,#t do
		return_t[i] = lnn.sumtable(t[i])
	end
	return return_t
end
```

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

   ### EXAMPLE
```lua
for i = 1,#weighted_layer_sums do
	nextlayer[i] = lnn.activation.sigmoid(weighted_layer_sums[i],false)
end
```

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="ta"></a>
   # lnn.activation.tanh()
   ### DESCRIPTION
   Returns `x` put into the hyperbolic tangent function if `derivative` is false and returns `x` put into the derivative of the hyperbolic tangent function if `derivative` is true.
   
   ### PARAMETERS
   | parameter name   | type        | description                                                                   |
   |------------------|-------------|-------------------------------------------------------------------------------|
   | **x**            | **number**  | The number you want to put into the tanh function.                            |
   | **derivative**   | **boolean** | Should the function return x put into the derivative of the tanh function.    |
   
   If `derivative` is false this function returns `x` put into the tanh (hyperbolic tangent) activation function, otherwise if `derivative` is true it returns `x` put into the derivative of the tanh activation function.

   ### MATHEMATICAL FUNCTION
   $$\frac{e^{2x}-1}{e^{2x}+1}$$

   ### EXAMPLE
```lua
for i = 1,#weighted_layer_sums do
	nextlayer[i] = lnn.activation.tanh(weighted_layer_sums[i],false)
end
```

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

   ### EXAMPLE
```lua
for i = 1,#weighted_layer_sums do
	nextlayer[i] = lnn.activation.relu(weighted_layer_sums[i],false)
end
```

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
 
   ### EXAMPLE
```lua
for i = 1,#weighted_layer_sums do
	nextlayer[i] = lnn.activation.leakyrelu(weighted_layer_sums[i],false)
end
```

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
   
   ### EXAMPLE
```lua
for i = 1,#weighted_layer_sums do
	nextlayer[i] = lnn.activation.elu(weighted_layer_sums[i],false)
end
```

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

   ### EXAMPLE
```lua
for i = 1,#weighted_layer_sums do
	nextlayer[i] = lnn.activation.swish(weighted_layer_sums[i],false)
end
```

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

   ### EXAMPLE
```lua
for i = 1,#weighted_layer_sums do
	nextlayer[i] = lnn.activation.binarystep(weighted_layer_sums[i],false)
end
```

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

   ### EXAMPLE
```lua
for i = 1,#weighted_layer_sums do
	nextlayer[i] = lnn.activation.leakyrelu(weighted_layer_sums[i],false)
end
```

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

   ### DATA CREATED IN _G[id]
```
_G[id] = {                 --TYPES:        DESCRIPTION:
	activation,            --string        activation function for neural network, in lnn.activation
	layercount,            --number        amount of hidden layers
	outcount,              --number        output size
	insize,                --number        input size
	alpha,                 --number        multiplication constant for some activation functions
	gradient = {           --table         gradient (actually calculated in lnn.adjust() or lnn.returngradient())
		gradw,             --table         (see lnn.returngradient() documentation)
		gradb,             --table         (see lnn.returngradient() documentation)
		gwsum,             --number        (see lnn.returngradient() documentation)
		gbsum,             --number        (see lnn.returngradient() documentation)
		dwinsum,           --number        (see lnn.returngradient() documentation)
		learningrate       --number        (see lnn.returngradient() documentation)
	},
	id,                    --string        id for the neural network
	weight,                --table         table containing tables containing weights
	bias,                  --table         table containing tables containing biases
	current,               --table         table containing tables containing node values
	layersizes             --table         sizes of every layer
}
```

   ### RETURN VALUE
   Nothing

   ### EXAMPLE
```lua
lnn.initialize("examplenn","leakyrelu",{16,20,15,10,5})
--[[
this neural network would have an id of "examplenn", the activation function would be "lnn.activation.leakyrelu()" and these would be the layer sizes:

input:            16
1st hidden layer: 20
2nd hidden layer: 15
3rd hidden layer: 10
output:           5
]]--
```

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

   Gets the output of `id` with input `intable`, go to SECTION THAT IDK NAME OF BUT FUTURE ME CHANGE THSI PSLS!!!!!! for information on how this function works.

   ### RETURN VALUE
   Neural network output (`_G[id]["current"][_G[id]["layercount"]+1]`)

   ### EXAMPLE
```lua
local inp = {
	0,1,1,0,
	1,1,1,1,
	1,1,1,1,
	0,1,1,0
}
local out = lnn.forwardpass("examplenn",inp)
```

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="ad"></a>
   # lnn.adjust()
   ### DESCRIPTION
   Calculates the gradient of neural network `id` with input `intable`, output `output` and expected/ideal output `expectedoutput` and adjusts `id`'s weights and biases.

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

   ### EXAMPLE
```lua
local eo = {0,0.25,0.5,0.75,1}
lnn.adjust("examplenn",inp,o,eo,0.01)
```

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="rg"></a>
   # lnn.returngradient()
   ### DESCRIPTION
   Calculates the gradient of neural network `id` with input `intable`, output `output` and expected/ideal output of `expectedoutput` and returns the gradient and any data that might be needed.

   ### PARAMETERS
   | parameter name     | type       | description                                                                                  |
   |--------------------|------------|----------------------------------------------------------------------------------------------|
   | **id**             | **string** | The id of the neural network you want to calculate the gradient of.                          |
   | **intable**        | **table**  | Input for the neural network.                                                                |
   | **output**         | **table**  | The real output for the neural network.                                                      |
   | **expectedoutput** | **table**  | The expected/ideal output for the neural network with an input of `intable`.                 |
   | **learningrate**   | **number** | THe step size for adjusting the weights and biases, usually a low number like 0.01 or 0.001. |

   ### GRADIENT TABLE
```
grad = {                   --TYPES:        DESCRIPTION:
	gradw,                 --table         weight gradient for output nodes
	gradb,                 --table         bias gradient for output nodes
	gwsum,                 --number        sum of weight gradient for hidden layer nodes
	gbsum,                 --number        sum of bias gradient for hidden layer nodes
	dwinsum,               --number        derivative of the weighed sum of the input
	learningrate           --number        step size
}
```

   ### RETURN VALUE
   Gradient

   ### EXAMPLE
```lua
local eo = {0,0.25,0.5,0.75,1}
local eo2 = {1,0.75}
local g1 = lnn.returngradient("examplenn",inp,o,eo,0.01)
local g2 = lnn.returngradient("examplenn",inp2,o2,eo2,0.01)
local g = {
	["gradw"] = {},
	["gradb"] = {},
	["gwsum"] = (g1["gwsum"]+g2["gwsum"])/2
	["gbsum"] = (g1["gbsum"]+g2["gbsum"])/2
	["dwinsum"] = (g1["dwinsum"]+g2["dwinsum"])/2
}
for i = 1,#g1["gradw"] do
	g["gradw"][i] = (g1["gradw"][i]+g2["gradw"][i])/2
	g["gradb"][i] = (g1["gradb"][i]+g2["gradb"][i])/2
end
```

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

   ### EXAMPLE
```lua
print(string.format("MSE (Mean Squared Error) of o and eo: %s",lnn.loss.mse(o,eo)))
```

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

   ### EXAMPLE
```lua
print(string.format("MAE (Mean Absolute Error) of o and eo: %s",lnn.loss.mae(o,eo)))
```

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

   ### EXAMPLE
```lua
print(string.format("SSE (Sum of Squared Error) of o and eo: %s",lnn.loss.sse(o,eo)))
```

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

   ### EXAMPLE
```lua
print(string.format("RMSE (Root of Mean Squared Error) of o and eo: %s",lnn.loss.rmse(o,eo)))
```

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

   ### EXAMPLE
```lua
print(string.format("Cross Entropy of o and eo: %s",lnn.loss.crossentropy(o,eo)))
```

</div>

## DEBUG FUNCTIONS

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="rw"></a>
   # lnn.debug.returnweights()
   ### DESCRIPTION
   Returns `id`'s weights.

   ### PARAMETERS
   | parameter name | type       | description                          |
   |----------------|------------|--------------------------------------|
   | **id**         | **string** | The id that you want the weights of. |

   Returns a table containing the weights for each layer.

   ### EXAMPLE
```lua
local weights = lnn.returnweights("examplenn")
```

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="rb"></a>
   # lnn.debug.returnbiases()
   ### DESCRIPTION
   Returns `id`'s biases.

   ### PARAMETERS
   | parameter name | type       | description                         |
   |----------------|------------|-------------------------------------|
   | **id**         | **string** | The id that you want the biases of. |

   Returns a table of tables containing the biases for each layer.

   ### EXAMPLE
```lua
local biases = lnn.returnbiases("examplenn")
```

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="rc"></a>
   # lnn.debug.returncurrents()
   ### DESCRIPTION
   Returns `id`'s current/node values.

   ### PARAMETERS
   | parameter name | type       | description                             |
   |----------------|------------|-----------------------------------------|
   | **id**         | **string** | The id that you want the node value of. |

   Returns a table of tables containing the current/node values.

   ### EXAMPLE
```lua
local current = lnn.returncurrent("examplenn")
```

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="ci"></a>
   # lnn.debug.clearid()
   ### DESCRIPTION
   Clears an id.

   ### PARAMETERS
   | parameter name | type       | description                    |
   |----------------|------------|--------------------------------|
   | **id**         | **string** | The id that you want to clear. |

   Sets `_G[id]` to nil.

   ### EXAMPLE
```lua
lnn.debug.clearid("examplenn")
```

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="ra"></a>
   # lnn.debug.randomize()
   ### DESCRIPTION
   Randomizes `id`'s weights and biases.

   ### PARAMETERS
   | parameter name | type       | description                                                  |
   |----------------|------------|--------------------------------------------------------------|
   | **id**         | **string** | The id that you want to randomize the weights and biases of. |

   This function sets `id`'s weights and biases to a random number between -1 and 1.

   ### EXAMPLE
```lua
lnn.debug.randomize("examplenn")
```

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="ar"></a>
   # lnn.debug.addrandom()
   ### DESCRIPTION
   Adds a random amount between `upperlimit` and `lowerlimit` to `id`'s weights and biases.

   ### PARAMETERS
   | parameter name | type       | description                                                               |
   |----------------|------------|---------------------------------------------------------------------------|
   | **id**         | **string** | The id that you want to add a random amount to the weights and biases of. |

   This function adds a random amount between `upperlimit` and `lowerlimit` to `id`'s weights and biases, could be useful when training a stickman to walk or something like that.

   ### EXAMPLE
```lua
lnn.debug.addrandom("examplenn",-0.1,0.1)
```

</div>

## DATA FUNCTIONS

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

   ### EXAMPLE
```lua
lnn.data.exportdata("examplenn","examplenn_data.txt")
```

</div>

<br>

<div style="border-radius: 15px; border: 2px solid rgb(100,100,120); padding: 10px;">
   
   <a id="id"></a>
   # lnn.data.importdata()
   ### DESCRIPTION
   Imports `filename` as `id`

   ### PARAMETERS
   | parameter name | type       | description                                   |
   |----------------|------------|-----------------------------------------------|
   | **id**         | **string** | The id that you want to export.               |
   | **filename**   | **string** | The name of the file you want to import from. |

   This function imports the data from `filename` to `_G[id]` if `id` is nil before importing.

   ### EXAMPLE
```lua
lnn.data.importdata("examplenn","examplenn_data.txt")
```

</div>
