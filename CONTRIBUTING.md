# Contribution Guidelines

## Table of Contents

 - [Code of Conduct](#code-of-conduct)
 - [I Have a Question](#i-have-a-question)
 - [I Want to Contribute](#i-want-to-contribute)
   - [Reporting Bugs](#reporting-bugs)
   - [Suggesting Enhancements](#suggesting-enhancements)
 - [Styleguide](#style-guide)
 - [Other Ways to Contribute](#other-ways-to-contribute)

## Code of Conduct

This project and everyone participating in it is governed by the [Synapsea Code of Conduct](https://github.com/mochji/synapsea/blob/stable/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [mochji](mailto:owo.mochji@gmail.com).

## I Have a Question

> If you want to ask a question, please make sure that you have read the [Documentation](https://sites.google.com/view/synapsea/api) first! :)

Before you ask a question, please make sure there are no existing [Issues](https://github.com/mochji/synapsea/issues) that might help you. In case you have found a suitable issue and still need clarification, you can write your question in said issue.

If you then still feel the need to ask a question and need clarification, we recommend the following:

 - Open an [Issue](https://github.com/mochji/synapsea/issues/new).
 - Provide as much context as you can about what you're running into.
 -

We will then take care of the issue as soon as possible.

## I Want To Contribute

> ### Notice
> When contributing to this project, you must follow the [GNU GPL v3](https://www.gnu.org/licenses/gpl-3.0.en.html).

### Reporting Bugs

#### Before Submitting a Bug Report

Make sure your bug report includes as much information as possible so others aren't needing to ask for more information. Therefore, please make sure you investigate the issue you're experiencing carefully. Please complete the following steps in advance to help fix a potential bug:

 - Make sure that you are using the latest version.
 - Determine if your bug is really a bug and not an error on your side and make sure that you have read the [documentation](https://sites.google.com/view/synapsea/api). If you are looking for support, you might want to check [this section](#i-have-a-question).
 - To see if other users have experienced (and potentially already solved) the same issue you are having, check if there is not already a bug report existing for your bug or error in the [bug tracker](https://github.com/mochji/synapsea/issues?q=label%3Abug).
 - Collect information about the bug:
   - Stack trace (Traceback)
   - OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
   - Version of the interpreter, compiler, SDK, runtime environment, package manager, depending on what seems relevant.
   - Possibly your input and the output
   - Can you reliably reproduce the issue? And can you also reproduce it with older versions?

#### How Do I Submit a Good Bug Report?

We use GitHub issues to track bugs and errors. If you run into an issue with the project:

 - Open an [Issue](https://github.com/mochji/synapsea/issues/new).
 - Add the relevant tags.
 - Explain the behavior you would expect and the actual behavior.
 - Please provide as much context as possible and describe the reproduction steps that someone else can follow to recreate the issue on their own. This usually includes your code. For good bug reports you should isolate the problem and create a reduced test case.
 - Provide the information you collected in the previous section.

Once it's filed:

 - A maintainer will try to reproduce the issue with your provided steps. If there are no reproduction steps or no obvious way to reproduce the issue, a maintainer will ask you for those steps and mark the issue as `needs repro`. Bugs with the `needs repro` tag will not be addressed until they are reproduced.
 - If a maintainer is able to reproduce the issue, it will be marked `needs fix`, as well as possibly other tags, and the issue will be worked on by a maintainer or a contributor.

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for Synapsea, **including completely new features and minor improvements to existing functionality**. Following these guidelines will help maintainers and the community to understand your suggestion and find related suggestions.

#### Before Submitting an Enhancement

 - Make sure that you are using the latest version.
 - Read the [documentation](https://sites.google.com/view/synapsea/api) carefully and find out if the functionality is already covered, maybe by an individual configuration.
 - Perform a [search](https://github.com/mochji/synapsea/issues) to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.
 - Make sure your idea fits within the scope and aims of the project. You should make a strong case to convice the developers that this feature should be added. If your idea only benefits a pretty small number of users, consider making it into a addon or plug-in library.

#### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues](https://github.com/mochji/synapsea/issues).

 - Use a **clear and descriptive title** for the issue to identify the suggestion.
 - Provide a **step-by-step description of the suggested enhancement** in as many details as possible.
 - **Describe the current behavior** and **explain which behavior you expected to see instead** and why. At this point you can also tell which alternatives do not work for you.
 - You may want to **include screenshots and animated GIFs** which help you demonstrate the steps or point out the part which the suggestion is related to.
 - **Explain why this enhancement would be useful** to most Synapsea users. You may also want to point out the other projects that solved it better and which could serve as inspiration.

## Styleguide

### Style

#### Whitespace

##### 1.0.0

 - Limit lines to 120 characters long.

##### 1.0.1

 - Use tabs for each indentation level and set your editor to display tabs as 4 characters.

##### 1.0.2

 - Include a whitespace before and after mathematical operators.

```lua
print(a + b)
```

##### 1.0.3

 - Include a whitespace after commas but not before.

```lua
local myTable = {1, 2}
```

##### 1.0.4

 - Do not have trailing commas.

```lua
--[[
    electric boogalo (not boogaloo) is an inside joke that only me and david (if you're reading this (most likely not)
    then you know who you are).
]]--

local myTable2ElectricBoogalo = {
    something + 2,
    something2 + 3
}
```

##### 1.0.5

 - Do not have trailing whitespace at the end of lines.

##### 1.0.6

 - **Do not** use DOS/Windows newlines (LF `\n` instead of CRLF `\r\n`).

##### 1.0.7

 - Blank lines should separate logical sections of the code.

```lua
local function flatten(tbl)
    local flattenFunc

    flattenFunc = function(tbl)
        local output = {}

        for a = 1, #tbl do
            if type(tbl[a]) == "table" then
                local flattenedInner = flattenFunc(tbl[a])

                for b = 1, #flattenedInner do
                    output[#output + 1] = flattenedInner[b]
                end
            else
                output[#output + 1] = tbl[a]
            end
        end

        return output
    end

    return flattenFunc(tbl)
end
```

##### 1.0.8

 - Minimize nesting as much as possible.
 - [Also watch this!](https://www.youtube.com/watch?v=CFRhGnuXG-4)

```lua
local function softplusActivation(x, derivative, alpha)
    if derivative then
        return 1 / (1 + math.exp(-x))
    end

    return math.log(1 + math.exp(x))
end
```

#### Naming

##### 1.1.0

 - Use descriptive and meaningful names for variables and functions.

##### 1.1.1

 - Use `camelCase` for local/mutable names.

##### 1.1.2

 - Constants should be in `CAPITALS_WITH_UNDERSCORES`.

##### 1.1.3

 - Use `lowercase` for utility function names in the `core/utils` directory.

##### 1.1.4

 - Number loop variables should start at `a` and decrement on each nested loop.

```lua
if #you.parents > 1 and you.parents.get(life.parent.type.MOTHER) then
    for a = 1, you.parents.get(life.parent.type.MOTHER).size do
        for b = 1, 10 do
            print(a, b) -- ?????????????????
        end
    end
end
```

##### 1.1.5

 - `in pairs` loop variables should describe what the variable is.

```lua
for parameter, parameterName in pairs(layer.parameters) do
    print("blah blah blah")
end
```

#### Exceptions

##### 1.2.0

 - Errors should follow the style of Lua errors.

#### Comments

##### 1.3.0

 - Use comments sparingly.
 - [Also watch this!](https://www.youtube.com/watch?v=Bf7vDBBOBUA)

##### 1.3.1

 - Comments should explain why the code is written a certain way, not what it does.

##### 1.3.2

 - Include a single space after the initial `--`.

##### 1.3.3

 - Use single-line comments `--` for single-line comments and block comments `--[[ ... ]]--` for multiline/block comments.

#### Tokens

##### 1.4.0

 - Include parenthesis when calling functions unless [see rule 1.4.1].

```lua
myFunction("wasd")
```

##### 1.4.1

 - For named arguments with a single arguments table, replace the parenthesis with braces.

```lua
myFunction2{
    input = 2,
    tired = true
}
```

##### 1.4.2

 - Use double quotes `"` for strings.

##### 1.4.3

 - Use single quotes `'` for characters unless the input can be more than 1 characer long (like io.open(fileName, HERE)).

### Logic

#### Functions

##### 2.0.0

 - Functions should do 1 - 2 concepts.

```lua
local function newModel(inputShape, metaData)
    -- blah blah blah
end

local function buildModel(model)
    -- nerd stuff
end
```

##### 2.0.1

 - Functions should not be absurdly wrong.

##### 2.0.2

 - Only abstract after 3 repeats to avoid coupling.

#### Variables

##### 2.1.0

 - Declare variables at the lowest scope possible.

##### 2.1.1

 - **Do not** use global variables.

##### 2.1.2

 - **Do not** have a global state.

#### Exceptions

##### 2.2.0

 - Use specific exceptions.

##### 2.2.1

 - Provide information about the error.

##### 2.2.2

 - Handle errors at an appropriate place in the code.

##### 2.2.3

 - Use `assert` instead of `if not x then error() end`.

### Best practices

##### 3.0.0

 - Avoid magic numbers.

##### 3.0.1

 - Write modular and expandable code.

##### 3.0.2

 - Periodically review and refactor code for readability and maintainability to avoid software rot.

## Other ways to contribute

All types of contributions are encouraged and valued. See the [Table of Contents](#table-of-contents) for different ways to help and details about how the project handles them. Please make sure to read the relevant section before making your contribution. It will make it a lot easier for us maintainers and smooth out the experience for all involved. The community looks forward to your contributions.

> And if you like Synapsea, but don't have the time to contribute, that's fine! There are other easy ways to support this project which we would be very happy about:
> - Star this repository
> - Mention Synapsea in your projects that use it
> - Just using Synapsea!

## Attribution

This guide is based on the [**contributing-gen**](https://github.com/bttger/contributing-gen).
