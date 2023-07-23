For now this is a todo list, make into README when code is done.

## FILES
###### (in core directory)

Filename           | Status   |
-------------------|----------|
../init.lua        | not done |
activation.lua     | done     |
backprop.lua       | not done |
initialize.lua     | done     |
layer.lua          | done?    |
layerbuild.lua     | not done |
loss.lua           | done     |
model.lua          | not done |
optimization.lua   | not done |
regularization.lua | not done |
syndebug.lua       | done     |
synmath.lua        | done     |
syntable.lua       | done     |
syntensor.lua      | not done |

## GENERAL TODO'S
###### (in order of what to do first)

 - Think of better structure for models.

## FILE SPECIFIC TODO'S
###### (in order of what to do first)

### syntensor.lua

 - Create general tensor structure.

 - Test speed on tensors. (If tensors are slower, just use regular tables.)

 - Create rest of the tensor functions.

### model.lua

 - If tensors are faster, use tensors instead but also check speed. If tensors with layers are slower, just use regular tables.)A

### layerbuild.lua

 - Create better way of storing parameters before initialization.

 - Finish creating `layerbuild` functions for all layers.

 - Use `syntensor` instead of `syntable`.

### model.lua

 - Rewrite model functions as they were just for testing.

 - Rewrite initialization to match new `layerbuild` parameter storing before initialization.

 - Model summary

### optimization.lua

 - Create more optimizers.

### regularization.lua

 - Create more regularizers.

### backprop.lua

 - Create the back-propagation functions for all the layers.

### ../init.lua

 - Finish importing the core files.

 - Write the functions for forward pass, backward pass, etc.
