--READ THE README.md FILE FOR DOCUMENTATION AND OTHER STUFF!!!!

_G["lnn"] = {
	["activation"] = {},
	["adjust"] = {
		["basic"] = {},
		["momentum"] = {}
	},
	["loss"] = {},
	["data"] = {}
}

--error catching/useful functions

function lnn.asserttype(variable,variablename,expectedtype)
	--check for errors in the function that checks for errors.
	if type(expectedtype)  ~= "string" or type(variablename) ~= "string" then
		error("variablename and thetype must be a string!")
	end

	--give an error if false or nil
	if type(variable) ~= expectedtype then
		error(string.format("%s (%s) is not a %s or is nil. Type: %s", variablename, tostring(variable), expectedtype, type(variable)))
	end
end

function lnn.assertsize(a,b,aname,bname,zerocheck)
	--check for errors in the function that checks for errors but different.
	if type(a) ~= "table" or type(b) ~= "table" then
		error("a and b must be a table!")
	end
	if type(aname) ~= "string" or type(bname) ~= "string" then
		error("aname and bname must be a string!")
	end

	--give an error they're not the same size or 0.
	if #a ~= #b then
		error(string.format("%s (%s) is not the same size as %s (%s).",aname,#a,bname,#b))
	end

	if zerocheck then
		if #a == 0 or #b == 0 then
			error(string.format("%s (%s) or %s (%s) is equal to zero.",aname,#a,bname,#b))
		end
	end
end

function lnn.findintable(item,table,recursive)
	--check for errors
	lnn.asserttype(table,"table","table")
	lnn.asserttype(recursive,"recursive","boolean")

	--do the stuff
	for i,v in pairs(table) do
		if table[i] == item then
			return i
		end
		if type(table[i]) == "table" and recursive then
			local returnval = lnn.findintable(item,table[i],true)
			if returnval then
				if type(returnval) == "table" then --this small detail took forever to get working
					for a = #returnval,1,-1 do
						returnval[a+1] = returnval[a]
					end
					returnval[1] = i
					return returnval
				end
				return {i,returnval}
			end
		end
	end
	return false
end

function lnn.sumtable(table)
	--check for errors
	lnn.asserttype(table,"table","table")

	--declare the variables
	local sum = 0

	--do the stuff
	for i = 1,#table do
		sum = sum + table[i]
	end

	return sum
end

--default activation functions

function lnn.activation.sigmoid(x,derivative)
	--check for errors
	lnn.asserttype(x,"x","number")
	lnn.asserttype(derivative,"derivative","boolean")

	--do the stuff
	if derivative then
		return (1 / (1 + math.exp(-x))) * (1-(1 / (1 + math.exp(-x))))
	else
		return 1 / (1 + math.exp(-x))
	end
end

function lnn.activation.tanh(x,derivative)
	--check for errors
	lnn.asserttype(x,"x","number")
	lnn.asserttype(derivative,"derivative","boolean")

	--do the stuff
	if derivative then
		return 1 - (((math.exp(2*x) - 1)/(math.exp(2*x) + 1))^2)
	end

	return (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
end

function lnn.activation.relu(x,derivative)
	--check for errors
	lnn.asserttype(x,"x","number")
	lnn.asserttype(derivative,"derivative","boolean")

	--do the stuff
	if derivative then
		if x > 0 then
			return 1
		end

		return 0
	end

	return math.max(0,x)
end

function lnn.activation.leakyrelu(x,derivative,alpha)
	--check for errors
	lnn.asserttype(x,"x","number")
	lnn.asserttype(derivative,"derivative","boolean")

	--do the stuff
	if derivative then
		if x > 0 then
			return 1
		end

		return alpha
	end

	if x > 0 then
		return x
	end

	return x*alpha
end

function lnn.activation.elu(x,derivative,alpha)
	--check for errors
	lnn.asserttype(x,"x","number")
	lnn.asserttype(derivative,"derivative","boolean")

	--do the stuff
	if derivative then
		if x < 0 then
			return alpha*math.exp(x)
		end

		return 1
	end

	if x < 0 then
		return alpha*(math.exp(x)-1)
	end

	return x
end

function lnn.activation.swish(x,derivative,alpha)
	--check for errors
	lnn.asserttype(x,"x","number")
	lnn.asserttype(derivative,"derivative","boolean")
	lnn.asserttype(alpha,"alpha","number")

	--do the stuff
	if derivative then
		return x/(1+math.exp(-alpha*x))+(1/1+math.exp(-x))*(1-x/(1+math.exp(-alpha*x)))
	end

	return x/(1+math.exp(-alpha*x))
end

function lnn.activation.binarystep(x,derivative)
	--check for errors
	lnn.asserttype(x,"x","number")
	lnn.asserttype(derivative,"derivative","boolean")

	--do the stuff
	if derivative then
		return 0
	end

	if x > 0 then
		return 1
	end

	return 0
end

function lnn.activation.softmax(x,derivative)
	--check for errors
	lnn.asserttype(x,"x","table")
	lnn.asserttype(derivative,"derivative","boolean")

	--declare the variables
	local expsum = 0
	local returntable = {}

	--do the stuff
	for i = 1,#x do
		expsum = expsum + math.exp(x[i])
	end

	if derivative then
		for i = 1,#x do
			returntable[i] = (math.exp(x[i])/expsum)*(1-(math.exp(x[i])/expsum))
		end

		return returntable
	end

	for i = 1,#x do
		returntable[i] = math.exp(x[i])/expsum
	end

	return returntable
end

function lnn.activation.softplus(x,derivative)
	--check for errors
	lnn.asserttype(x,"x","number")
	lnn.asserttype(derivative,"derivative","boolean")

	--do the stuff
	if derivative then
		return math.exp(x)/(1+math.exp(x)) --this function's derivative is the exact same as the sigmoid activation function
	end

	return math.log(1+math.exp(x))
end

function lnn.activation.softsign(x,derivative)
	--check for errors
	lnn.asserttype(x,"x","number")
	lnn.asserttype(derivative,"derivative","boolean")

	--do the stuff
	if derivative then
		if x == 0 then return 1 end --hell yeah that only took me 2 minutes to come up with that solution!
		return x/(x*(1+math.abs(x)^2))
	end

	return x/1+math.abs(x)
end

function lnn.activation.linear(x,derivative,alpha)
	--check for errors
	lnn.asserttype(x,"x","number")
	lnn.asserttype(derivative,"derivative","boolean")
	lnn.asserttype(alpha,"alpha","number")

	--do the stuff
	if derivative then
		return alpha
	end

	return x*alpha
end

--neural network functions

function lnn.initialize(id,activation,layersizes)
	--check for errors
	lnn.asserttype(id,"id","string")
	lnn.asserttype(activation,"activation","string")
	lnn.asserttype(layersizes,"layersizes","table")

	--check if the id already exists
	if _G[id] ~= nil then
		error(string.format("id (%s) already exists, use the 'lnn.debug.clearid()' to clear the id.",id))
	end

	--check if layersizes has less than 2 items
	if #layersizes < 2 then
		error("layersizes must have at least 2 items for input size and output size. size of layersizes: "..#layersizes)
	end

	--check if the activation is a valid activation function
	if type(lnn.activation[activation]) ~= "function" then
		error(string.format("%s was not found in lnn.activation, available default activation functions are: 'sigmoid', 'tanh', 'relu', 'leakyrelu', 'elu', 'swish', 'binarystep', 'softplus' and 'linear'.",activation))
	end

	--initialize the neural network

	--initialize the neural network data
	_G[id] = {
		["activations"] = {},
		["alpha"] = 1,
		["gradient"] = {
			["error"] = {},
			["grad"] = {
				["bias"] = {},
				["weight"] = {}
			},
			["intable"] = {},
			["learningrate"] = 0,
			["momentum"] = 1
		},
		["id"] = id,
		["weight"] = {},
		["bias"] = {},
		["current"] = {},
		["layersizes"] = layersizes,
		["weightcount"] = 0
	}

	if activation == "leakyrelu" then
		_G[id]["alpha"] = 0.01
	end

	--initialize the neural network layers

	local amounttofill = 0

	--create the tables
	for a = 1,#layersizes-1 do
		--add into activations
		_G[id]["activations"][a] = activation

		_G[id]["current"][a] = {}
		_G[id]["weight"][a] = {}

		--create the tables for the current values
		for i = 1,layersizes[a+1] do
			_G[id]["current"][a][i] = 0.0
		end

		--calculate amounttofill
		amounttofill = layersizes[a+1]*layersizes[a]

		--create the tables for the connection values (weight)
		for i = 1,amounttofill do
			_G[id]["weight"][a][i] = math.random(-100,100)/100
		end

		--add to weightcount
		_G[id]["weightcount"] = _G[id]["weightcount"] + amounttofill
	end

	for i = 1,#layersizes-2 do
		_G[id]["bias"][i] = math.random(-100,100)/100 --yay im stupid, i thought each node had it's own bias, nope! every layer has a bias that is added to each node so each node has a bias, just not a unique bias. luckily, i don't have to change the code much!
	end
end

function lnn.forwardpass(id,intable)
	--check for errors
	lnn.asserttype(id,"id","string")
	lnn.asserttype(intable,"intable","table")

	--check if the id doesn't exist
	if _G[id] == nil then
		error(string.format("id (%s) doesnt exist.",id))
	end

	if #intable ~= _G[id]["layersizes"][1] then
		error(string.format("intable (%s) is not the same size as the input size when id (%s) was initialized (%s).",#intable,id,_G[id]["layersizes"][1]))
	end

	--declare the functions
	local function getlayer(lastlayer,nextlayer,weights,bias,layernum) --i am super proud of this function :D
		--declare the variables
		local sum = 0

		--get the sum of the connected weights to the current node we are on and replace the nextlayer
		for a = 1,#nextlayer do
			for i = 1,#lastlayer do
				sum = sum + lastlayer[i]*(weights[i+((a-1)*#lastlayer)])
			end
			nextlayer[a] = lnn.activation[_G[id]["activations"][layernum]](sum+bias,false,_G[id]["alpha"]) --even if it only has 2 parameters this still works but it pains me to do that
			sum = 0
		end
	end

	--do the stuff
	getlayer(intable,_G[id]["current"][1],_G[id]["weight"][1],0,1) --input layer to first hidden or output
	for i = 2,#_G[id]["layersizes"]-1 do --rest of the hidden layers and output
		getlayer(_G[id]["current"][i-1],_G[id]["current"][i],_G[id]["weight"][i],_G[id]["bias"][i-1],i)
	end

	return _G[id]["current"][#_G[id]["current"]] --last table in current is output layer
end

function lnn.returnerror(id,intable,output,expectedoutput,learningrate)
	--check for errors
	lnn.asserttype(id,"id","string")
	lnn.asserttype(intable,"intable","table")
	lnn.asserttype(output,"output","table")
	lnn.asserttype(expectedoutput,"expectedoutput","table")
	lnn.asserttype(learningrate,"learningrate","number")
	lnn.assertsize(output,expectedoutput,"output","expectedoutput",true)

	if _G[id] == nil then
		error(string.format("id (%s) doesn't exist.",id))
	end

	if #intable ~= _G[id]["layersizes"][1] then
		error(string.format("intable (%s) is not the same size as the input size when id (%s) was initialized (%s).",#intable,id,_G[id]["layercount"][1]))
	end

	--declare the variables
	local err = {{}}

	--calculate the error for each output node
	for i = 1,#output do
		err[1][i] = (output[i]-expectedoutput[i])*lnn.activation[_G[id]["activations"][#_G[id]["activations"]]](output[i],true,_G[id]["alpha"])+(output[i]-expectedoutput[i])*learningrate
	end

	--backpropagate the error
	for a = #_G[id]["layersizes"]-2,1,-1 do
		err[#err+1] = {}
		for b = 1,_G[id]["layersizes"][a] do
			err[#err][b] = 0
			for i = 1,_G[id]["layersizes"][a+1] do
				err[#err][b] = err[#err][b] + (_G[id]["weight"][a][i+((b-1)*_G[id]["layersizes"][a])]*err[#err-1][math.ceil(b/(_G[id]["layersizes"][a]/#err[#err-1]))])*lnn.activation[_G[id]["activations"][a]](_G[id]["current"][a][b],true,_G[id]["alpha"]) --you never saw this.
			end
		end
	end

	return err
end

function lnn.adjust.adjustfromgradient(id,gradient)
	--check for errors
	lnn.asserttype(id,"id","string")
	lnn.asserttype(gradient,"gradient","table")

	if _G[id] == nil then
		error(string.format("id (%s) doesn't exist.",id))
	end

	if #gradient["intable"] ~= _G[id]["layersizes"][1] then
		error(string.format("intable (%s) is not the same size as the input size when id (%s) was initialized (%s).",#gradient["intable"],id,_G[id]["layercount"][1]))
	end

	--update the weights
	for a = #_G[id]["layersizes"]-1,1,-1 do
		for i = 1,#_G[id]["weight"][a] do
			_G[id]["weight"][a][i] = _G[id]["weight"][a][i] - gradient["grad"]["weight"][#_G[id]["layersizes"]-a][i]
		end
	end

	--update the biases
	for i = #_G[id]["layersizes"]-2,1,-1 do
		_G[id]["bias"][i] = _G[id]["bias"][i] - gradient["grad"]["bias"][i]
	end

	--update the data on _G[id]
	_G[id]["gradient"] = gradient
end

--basic adjust functions

function lnn.adjust.basic.adjust(id,intable,output,expectedoutput,learningrate)
	--check for errors
	lnn.asserttype(id,"id","string")
	lnn.asserttype(intable,"intable","table")
	lnn.asserttype(output,"output","table")
	lnn.asserttype(expectedoutput,"expectedoutput","table")
	lnn.asserttype(learningrate,"learningrate","number")
	lnn.assertsize(output,expectedoutput,"output","expectedoutput",true)

	if _G[id] == nil then
		error(string.format("id (%s) doesn't exist.",id))
	end

	if #intable ~= _G[id]["layersizes"][1] then
		error(string.format("intable (%s) is not the same size as the input size when id (%s) was initialized (%s).",#intable,id,_G[id]["layercount"][1]))
	end

	--declare the variables
	local grad = {
		["error"] = {{}},
		["grad"] = {
			["bias"] = {},
			["weight"] = {{}}
		},
		["intable"] = intable,
		["learningrate"] = learningrate,
		["momentum"] = _G[id]["gradient"]["momentum"]
	}

	--calculate the error for each output node
	for i = 1,#output do
		grad["error"][1][i] = (output[i]-expectedoutput[i])*lnn.activation[_G[id]["activations"][#_G[id]["activations"]]](output[i],true,_G[id]["alpha"])+(output[i]-expectedoutput[i])*learningrate
	end

	--backpropagate the error
	for a = #_G[id]["layersizes"]-2,1,-1 do
		grad["error"][#grad["error"]+1] = {}
		for b = 1,_G[id]["layersizes"][a+1] do
			grad["error"][#grad["error"]][b] = 0
			for i = 1,_G[id]["layersizes"][a] do
				grad["error"][#grad["error"]][b] = grad["error"][#grad["error"]][b] + (_G[id]["weight"][a][i+((b-1)*_G[id]["layersizes"][a])]*grad["error"][#grad["error"]-1][math.ceil(b/(_G[id]["layersizes"][a]/#grad["error"][#grad["error"]-1]))])*lnn.activation[_G[id]["activations"][a]](_G[id]["current"][a][b],true,_G[id]["alpha"]) --you never saw this.
			end
		end
	end

	--update the weights
	for a = #_G[id]["layersizes"]-1,1,-1 do
		for b = 1,_G[id]["layersizes"][a+1] do
			grad["grad"]["weight"][#_G[id]["layersizes"]-a] = {}
			for i = 1,_G[id]["layersizes"][a] do
				local curgrad = math.min(math.max((learningrate/(_G[id]["weightcount"]))*(grad["error"][#_G[id]["layersizes"]-a][b]*lnn.sumtable(intable)),-learningrate),learningrate) --exploding gradients are even more of a problem than i thought they were
				_G[id]["weight"][a][i+((b-1)*_G[id]["layersizes"][a])] = _G[id]["weight"][a][i+((b-1)*_G[id]["layersizes"][a])] - curgrad
				grad["grad"]["weight"][#_G[id]["layersizes"]-a][i+((b-1)*_G[id]["layersizes"][a])] = curgrad
			end
		end
	end

	--update the biases
	for i = #_G[id]["layersizes"]-2,1,-1 do
		local curgrad = math.min(math.max((learningrate/(#_G[id]["layersizes"]-2))*lnn.sumtable(grad["error"][(#_G[id]["layersizes"]-1)-i]),-learningrate),learningrate)
		_G[id]["bias"][i] = _G[id]["bias"][i] - curgrad
		grad["grad"]["bias"][i] = curgrad
	end

	--update the data on _G[id]
	_G[id]["gradient"] = grad
end

function lnn.adjust.basic.returngradient(id,intable,output,expectedoutput,learningrate)
	--check for errors
	lnn.asserttype(id,"id","string")
	lnn.asserttype(intable,"intable","table")
	lnn.asserttype(output,"output","table")
	lnn.asserttype(expectedoutput,"expectedoutput","table")
	lnn.asserttype(learningrate,"learningrate","number")
	lnn.assertsize(output,expectedoutput,"output","expectedoutput",true)

	if _G[id] == nil then
		error(string.format("id (%s) doesn't exist.",id))
	end

	if #intable ~= _G[id]["layersizes"][1] then
		error(string.format("intable (%s) is not the same size as the input size when id (%s) was initialized (%s).",#intable,id,_G[id]["layercount"][1]))
	end

	--declare the variables
	local grad = {
		["error"] = {{}},
		["grad"] = {
			["bias"] = {},
			["weight"] = {{}}
		},
		["intable"] = intable,
		["learningrate"] = learningrate,
		["momentum"] = _G[id]["gradient"]["momentum"]
	}

	--calculate the error for each output node
	for i = 1,#output do
		grad["error"][1][i] = (output[i]-expectedoutput[i])*lnn.activation[_G[id]["activations"][#_G[id]["activations"]]](output[i],true,_G[id]["alpha"])+(output[i]-expectedoutput[i])*learningrate
	end

	--backpropagate the error
	for a = #_G[id]["layersizes"]-2,1,-1 do
		grad["error"][#grad["error"]+1] = {}
		for b = 1,_G[id]["layersizes"][a+1] do
			grad["error"][#grad["error"]][b] = 0
			for i = 1,_G[id]["layersizes"][a] do
				grad["error"][#grad["error"]][b] = grad["error"][#grad["error"]][b] + (_G[id]["weight"][a][i+((b-1)*_G[id]["layersizes"][a])]*grad["error"][#grad["error"]-1][math.ceil(b/(_G[id]["layersizes"][a]/#grad["error"][#grad["error"]-1]))])*lnn.activation[_G[id]["activations"][a]](_G[id]["current"][a][b],true,_G[id]["alpha"]) --you never saw this.
			end
		end
	end

	--calculate the weight gradient
	for a = #_G[id]["layersizes"]-1,1,-1 do
		for b = 1,_G[id]["layersizes"][a+1] do
			grad["grad"]["weight"][#_G[id]["layersizes"]-a] = {}
			for i = 1,_G[id]["layersizes"][a] do
				grad["grad"]["weight"][#_G[id]["layersizes"]-a][i+((b-1)*_G[id]["layersizes"][a])] = math.min(math.max((learningrate/(_G[id]["weightcount"]))*(grad["error"][#_G[id]["layersizes"]-a][b]*lnn.sumtable(intable)),-learningrate),learningrate)
			end
		end
	end

	--calculate the bias gradient
	for i = #_G[id]["layersizes"]-2,1,-1 do
		grad["grad"]["bias"][i] = math.min(math.max((learningrate/(#_G[id]["layersizes"]-2))*lnn.sumtable(grad["error"][(#_G[id]["layersizes"]-1)-i]),-learningrate),learningrate)
	end

	return grad
end


--momentum adjust functions

function lnn.adjust.momentum.adjust(id,intable,output,expectedoutput,learningrate)
	--check for errors
	lnn.asserttype(id,"id","string")
	lnn.asserttype(intable,"intable","table")
	lnn.asserttype(output,"output","table")
	lnn.asserttype(expectedoutput,"expectedoutput","table")
	lnn.asserttype(learningrate,"learningrate","number")
	lnn.assertsize(output,expectedoutput,"output","expectedoutput",true)

	if _G[id] == nil then
		error(string.format("id (%s) doesn't exist.",id))
	end

	if #intable ~= _G[id]["layersizes"][1] then
		error(string.format("intable (%s) is not the same size as the input size when id (%s) was initialized (%s).",#intable,id,_G[id]["layercount"][1]))
	end

	--declare the variables
	local grad = {
		["error"] = {{}},
		["grad"] = {
			["bias"] = {},
			["weight"] = {{}}
		},
		["intable"] = intable,
		["learningrate"] = learningrate,
		["momentum"] = _G[id]["gradient"]["momentum"]
	}
	local pweightdelta = 0
	local pbiasdelta = 0

	--calculate the error for each output node
	for i = 1,#output do
		grad["error"][1][i] = (output[i]-expectedoutput[i])*lnn.activation[_G[id]["activations"][#_G[id]["activations"]]](output[i],true,_G[id]["alpha"])+(output[i]-expectedoutput[i])*learningrate
	end

	--backpropagate the error
	for a = #_G[id]["layersizes"]-2,1,-1 do
		grad["error"][#grad["error"]+1] = {}
		for b = 1,_G[id]["layersizes"][a+1] do
			grad["error"][#grad["error"]][b] = 0
			for i = 1,_G[id]["layersizes"][a] do
				grad["error"][#grad["error"]][b] = grad["error"][#grad["error"]][b] + (_G[id]["weight"][a][i+((b-1)*_G[id]["layersizes"][a])]*grad["error"][#grad["error"]-1][math.ceil(b/(_G[id]["layersizes"][a]/#grad["error"][#grad["error"]-1]))])*lnn.activation[_G[id]["activations"][a]](_G[id]["current"][a][b],true,_G[id]["alpha"]) --you never saw this.
			end
		end
	end

	--update the weights
	for a = #_G[id]["layersizes"]-1,1,-1 do
		for b = 1,_G[id]["layersizes"][a+1] do
			grad["grad"]["weight"][#_G[id]["layersizes"]-a] = {}
			for i = 1,_G[id]["layersizes"][a] do
				local curgrad = math.min(math.max((learningrate/(_G[id]["weightcount"]))*(grad["error"][#_G[id]["layersizes"]-a][b]*lnn.sumtable(intable)),-learningrate),learningrate) --exploding gradients are even more of a problem than i thought they were

				--calculate the momentum
				local dweight = curgrad+(grad["momentum"]*learningrate)*pweightdelta
				grad["momentum"] = grad["momentum"] + (1-grad["momentum"])*curgrad

				--update the weight
				_G[id]["weight"][a][i+((b-1)*_G[id]["layersizes"][a])] = _G[id]["weight"][a][i+((b-1)*_G[id]["layersizes"][a])] - dweight
				grad["grad"]["weight"][#_G[id]["layersizes"]-a][i+((b-1)*_G[id]["layersizes"][a+1])] = dweight

				--store weight update
				pweightdelta = dweight
			end
		end
	end

	--update the biases
	for i = #_G[id]["layersizes"]-2,1,-1 do
		local curgrad = math.min(math.max((learningrate/(#_G[id]["layersizes"]-2))*lnn.sumtable(grad["error"][(#_G[id]["layersizes"]-1)-i]),-learningrate),learningrate)

		--calculate the momentum
		local dbias = curgrad+(grad["momentum"]*learningrate)*pbiasdelta --programming is possibly the most fufilling thing ive ever done
		grad["momentum"] = grad["momentum"] + (1-grad["momentum"])*curgrad

		--update the bias
		_G[id]["bias"][i] = _G[id]["bias"][i] - curgrad
		grad["grad"]["bias"][i] = curgrad

		--store bias update
		pweightdelta = dbias
	end

	--update the data on _G[id]
	_G[id]["gradient"] = grad
end

function lnn.adjust.momentum.returngradient(id,intable,output,expectedoutput,learningrate)
	--check for errors
	lnn.asserttype(id,"id","string")
	lnn.asserttype(intable,"intable","table")
	lnn.asserttype(output,"output","table")
	lnn.asserttype(expectedoutput,"expectedoutput","table")
	lnn.asserttype(learningrate,"learningrate","number")
	lnn.assertsize(output,expectedoutput,"output","expectedoutput",true)

	if _G[id] == nil then
		error(string.format("id (%s) doesn't exist.",id))
	end

	if #intable ~= _G[id]["layersizes"][1] then
		error(string.format("intable (%s) is not the same size as the input size when id (%s) was initialized (%s).",#intable,id,_G[id]["layercount"][1]))
	end

	--declare the variables
	local grad = {
		["error"] = {{}},
		["grad"] = {
			["bias"] = {},
			["weight"] = {{}}
		},
		["intable"] = intable,
		["learningrate"] = learningrate,
		["momentum"] = _G[id]["gradient"]["momentum"]
	}
	local pweightdelta = 0
	local pbiasdelta = 0

	--calculate the error for each output node
	for i = 1,#output do
		grad["error"][1][i] = (output[i]-expectedoutput[i])*lnn.activation[_G[id]["activations"][#_G[id]["activations"]]](output[i],true,_G[id]["alpha"])+(output[i]-expectedoutput[i])*learningrate
	end

	--backpropagate the error
	for a = #_G[id]["layersizes"]-2,1,-1 do
		grad["error"][#grad["error"]+1] = {}
		for b = 1,_G[id]["layersizes"][a+1] do
			grad["error"][#grad["error"]][b] = 0
			for i = 1,_G[id]["layersizes"][a] do
				grad["error"][#grad["error"]][b] = grad["error"][#grad["error"]][b] + (_G[id]["weight"][a][i+((b-1)*_G[id]["layersizes"][a])]*grad["error"][#grad["error"]-1][math.ceil(b/(_G[id]["layersizes"][a]/#grad["error"][#grad["error"]-1]))])*lnn.activation[_G[id]["activations"][a]](_G[id]["current"][a][b],true,_G[id]["alpha"]) --you never saw this.
			end
		end
	end

	--calculate the weight gradient
	for a = #_G[id]["layersizes"]-1,1,-1 do
		for b = 1,_G[id]["layersizes"][a+1] do
			grad["grad"]["weight"][#_G[id]["layersizes"]-a] = {}
			for i = 1,_G[id]["layersizes"][a] do
				local curgrad = math.min(math.max((learningrate/(_G[id]["weightcount"]))*(grad["error"][#_G[id]["layersizes"]-a][b]*lnn.sumtable(intable)),-learningrate),learningrate) --exploding gradients are even more of a problem than i thought they were

				--calculate the momentum
				grad["momentum"] = grad["momentum"] + (1-grad["momentum"])*curgrad

				grad["grad"]["weight"][#_G[id]["layersizes"]-a][i+((b-1)*_G[id]["layersizes"][a])] = curgrad+(grad["momentum"]*learningrate)*pweightdelta

				--store weight update
				pweightdelta = curgrad+(grad["momentum"]*learningrate)*pweightdelta
			end
		end
	end

	--calculate the bias gradient
	for i = #_G[id]["layersizes"]-2,1,-1 do
		local curgrad = math.min(math.max((learningrate/(#_G[id]["layersizes"]-2))*lnn.sumtable(grad["error"][(#_G[id]["layersizes"]-1)-i]),-learningrate),learningrate)

		--calculate the momentum
		grad["momentum"] = grad["momentum"] + (1-grad["momentum"])*curgrad

		grad["grad"]["bias"][i] = curgrad+(grad["momentum"]*learningrate)*pbiasdelta

		--store bias update
		pweightdelta = curgrad+(grad["momentum"]*learningrate)*pbiasdelta
	end

	return grad
end

--default loss functions

function lnn.loss.mse(output,expectedoutput)
	--check for errors
	lnn.asserttype(output,"output","table")
	lnn.asserttype(expectedoutput,"expectedutput","table")

	lnn.assertsize(output,expectedoutput,"output","expectedoutput")

	--declare the variables
	local mse = 0

	--do the stuff
	for i = 1,#output do
		mse = mse + (expectedoutput[i] - output[i])^2
	end
	return mse/#output
end

function lnn.loss.mae(output,expectedoutput)
	--check for errors
	lnn.asserttype(output,"output","table")
	lnn.asserttype(expectedoutput,"expectedutput","table")

	lnn.assertsize(output,expectedoutput,"output","expectedoutput")

	--declare the variables
	local mae = 0

	--do the stuff
	for i = 1,#output do
		mae = mae + math.abs(output[i] - expectedoutput[i])
	end
	return mae/#output
end

function lnn.loss.sse(output,expectedoutput)
	--check for errors
	lnn.asserttype(output,"output","table")
	lnn.asserttype(expectedoutput,"expectedutput","table")

	lnn.assertsize(#output,#expectedoutput,"output","expectedoutput")

	--declare the variables
	local sse = 0

	--do the stuff
	for i = 1,#output do
		sse = (output[i]-expectedoutput[i])^2
	end
	return sse/#output
end

function lnn.loss.rmse(output,expectedoutput)
	--check for errors
	lnn.asserttype(output,"output","table")
	lnn.asserttype(expectedoutput,"expectedutput","table")

	lnn.assertsize(output,expectedoutput,"output","expectedoutput")

	--declare the variables
	local rmse = 0

	--do the stuff
	for i = 1,#output do
		rmse = rmse + (expectedoutput[i] - output[i])^2
	end

	return math.sqrt((rmse/#output))
end

function lnn.loss.crossentropy(output,expectedoutput)
	--check for errors.
	lnn.asserttype(output,"output","table")
	lnn.asserttype(expectedoutput,"expectedoutput","table")

	lnn.assertsize(output,expectedoutput,"output","expectedoutput",true)

	--declare the variables
	local sum = 0

	--do the stuff
	for i = 1,#output do
		sum = sum + (expectedoutput[i]*math.log(output[i])) + (1-expectedoutput[i]) * math.log(1-output[i])
	end
	return -sum
end

function lnn.loss.categoricalcrossentropy(output,expectedoutput)
	--check for errors
	lnn.asserttype(output,"output","table")
	lnn.asserttype(expectedoutput,"expectedoutput","table")
	lnn.assertsize(output,expectedoutput,"output","expectedoutput")

	--declare the variables
	local sum = 0

	--do the stuff
	for i = 1,#output do
		sum = sum + (expectedoutput[i]*math.log(output[i]))
	end

	return -sum
end

--data functions, yayaya

function lnn.data.randomize(id)
	--check for errors
	lnn.asserttype(id,"id","string")

	if _G[id] == nil then
		error(string.format("id (%s) doesn't exist.",id))
	end

	--do the stuff

	for a = 1,#_G[id]["layersizes"]-1 do
		--randomize biases
		_G[id]["bias"][a] = math.random(-100,100)/100

		--randomize weights
		for i = 1,#_G[id]["weight"][a] do
			_G[id]["weight"][a][i] = math.random(-100,100)/100
		end
	end
end

function lnn.data.addrandom(id,lowerlimit,upperlimit)
	--check for errors
	lnn.asserttype(id,"id","string")
	lnn.asserttype(lowerlimit,"lowerlimit","number")
	lnn.asserttype(upperlimit,"upperlimit","number")

	if _G[id] == nil then
		error(string.format("id (%s) doesn't exist.",id))
	end

	--do the stuff
	for a = 1,#_G[id]["layersizes"]-1 do
		--randomize biases
		_G[id]["bias"][a] = _G[id]["bias"][a] + lowerlimit + math.random()*(upperlimit-lowerlimit) --https://stackoverflow.com/a/59494965 :D

		--randomize weights
		for i = 1,#_G[id]["weight"][a] do
			_G[id]["weight"][a][i] = _G[id]["weight"][a][i] + lowerlimit + math.random()*(upperlimit-lowerlimit)
		end
	end
end

--why is there only exportdata? because you can just use require() if its .lua or dofile() to import the data.

function lnn.data.exportdata(id,filename)
	--check for errors.
	lnn.asserttype(id,"id","string")
	lnn.asserttype(filename,"filename","string")

	if _G[id] == nil then
		error(string.format("id (%s) doesn't exist.",id))
	end

	--clear the file
	io.open(filename,"w"):close()

	--do the stuff
	local f = io.open(filename,"a+")

	if f ~= nil then
		--write the basic data
		f:write(string.format("return {\n['alpha'] = %s,\n['id'] = '%s',\n['activations'] = {",_G[id]["alpha"],id))

		--write the activations
		for i = 1,#_G[id]["activations"] do
			f:write(string.format("'%s'%s",_G[id]["activations"][i],","))
		end
		f:write("},\n['weight'] = {\n")

		--write the weight data
		for a = 1,#_G[id]["layersizes"]-1 do
			f:write("{")
			for i = 1,#_G[id]["weight"][a] do
				f:write(_G[id]["weight"][a][i]..",")
			end
			f:write("},\n")
		end

		f:write("},\n['current'] = {\n")

		--write the current data
		for a = 1,#_G[id]["layersizes"]-1 do
			f:write("{")
			for i = 1,#_G[id]["current"][a] do
				f:write(_G[id]["current"][a][i]..",")
			end
			f:write("},\n")
		end

		f:write("},\n['bias'] = {")

		--write the bias data
		for i = 1,#_G[id]["layersizes"]-2 do
			f:write(_G[id]["bias"][i]..",")
		end

		f:write("},\n['layersizes'] = {")

		--write layersizes
		for i = 1,#_G[id]["layersizes"] do
			f:write(_G[id]["layersizes"][i]..",")
		end

		--write weightcount
		f:write(string.format("},\n['weightcount'] = %s,\n",_G[id]["weightcount"]))

		--write the gradient data
		f:write("['gradient'] = {\n")

		f:write("['error'] = {")
		for a = 1,#_G[id]["gradient"]["error"] do
			f:write("\n{")
			for i = 1,#_G[id]["gradient"]["error"][a] do
				f:write(_G[id]["gradient"]["error"][a][i]..",")
			end
			f:write("},\n")
		end
		f:write("},\n")

		f:write("['grad'] = {\n['bias'] = {")
		for i = 1,#_G[id]["gradient"]["grad"]["bias"] do
			f:write(_G[id]["gradient"]["grad"]["bias"][i]..",")
		end
		f:write("},\n")

		f:write("['weight'] = {")
		for a = 1,#_G[id]["gradient"]["grad"]["weight"] do
			f:write("\n{")
			for i = 1,#_G[id]["gradient"]["grad"]["weight"][a] do
				f:write(_G[id]["gradient"]["grad"]["weight"][a][i]..",")
			end
			f:write("},\n")
		end

		f:write("}\n},\n['intable'] = {")

		for i = 1,#_G[id]["gradient"]["intable"] do
			f:write(_G[id]["gradient"]["intable"][i]..",")
		end

		f:write(string.format("},\n['learningrate'] = %s,\n['momentum'] = %s\n}\n}",_G[id]["gradient"]["learningrate"],_G[id]["gradient"]["momentum"]))
	else
		print("something went wrong, f is nil?")
	end
end

return lnn --its require() time
