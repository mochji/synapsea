# Contributing to Synapsea

Thank you so much for contributing to Synapsea! :D

All types of contributions are encouraged and valued. See the [Table of Contents](#table-of-contents) for different ways to help and details about how the project handles them. Please make sure to read the relevant section before making your contribution. It will make it a lot easier for us maintainers and smooth out the experience for all involved. The community looks forward to your contributions.

> If you like Synapsea, but just don't have time to contribute, that's fine! There are other easy ways to support the project and show your appreciation, which we would also be very happy about:
> - Star the project
> - Refer this project in your project's README
> - Mention Synapsea to people

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [I Have a Question](#i-have-a-question)
- [I Want To Contribute](#i-want-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
 - [Coding styleguide](#coding-style)

## Code of Conduct

This project and everyone participating in it is governed by the [Synapsea Code of Conduct](https://github.com/mochji/synapseablob/master/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [mochji](mailto:owo.mochji@gmail.com).

## I Have a Question

> If you want to ask a question, please make sure that you have read the [Documentation](https://sites.google.com/view/synapsea/api) first! :)

Before you ask a question, please make sure there are no existing [Issues](https://github.com/mochji/synapsea/issues) that might help you. In case you have found a suitable issue and still need clarification, you can write your question in said issue.

If you then still feel the need to ask a question and need clarification, we recommend the following:

- Open an [Issue](https://github.com/mochji/synapsea/issues/new).
- Provide as much context as you can about what you're running into.
- Provide project and platform versions (nodejs, npm, etc), depending on what seems relevant.

We will then take care of the issue as soon as possible.

## I Want To Contribute

> ### Notice
> When contributing to this project, you must agree that you follow this project's [License](https://www.gnu.org/licenses/gpl-3.0.en.html).

### Reporting Bugs

#### Before Submitting a Bug Report

Make sure your bug report includes as much information as possible so others aren't needing to ask for more information. Therefore, please make sure you investigate the issue you're experiencing carefully. Please complete the following steps in advance to help fix a potential bug:

- Make sure that you are using the latest version.
- Determine if your bug is really a bug and not an error on your side e.g. using incompatible environment components/versions (Make sure that you have read the [documentation](https://sites.google.com/view/synapsea/api). If you are looking for support, you might want to check [this section](#i-have-a-question)).
- To see if other users have experienced (and potentially already solved) the same issue you are having, check if there is not already a bug report existing for your bug or error in the [bug tracker](https://github.com/mochji/synapsea/issues?q=label%3Abug).
- Collect information about the bug:
  - Stack trace (Traceback)
  - OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
  - Version of the interpreter, compiler, SDK, runtime environment, package manager, depending on what seems relevant.
  - Possibly your input and the output
  - Can you reliably reproduce the issue? And can you also reproduce it with older versions?

#### How Do I Submit a Good Bug Report?

> You must never report security related issues, vulnerabilities or bugs including sensitive information to the issue tracker, or elsewhere in public. Instead sensitive bugs must be sent by email to [mochji](mailto:mochji@gmail.com).

We use GitHub issues to track bugs and errors. If you run into an issue with the project:

- Open an [Issue](https://github.com/mochji/synapsea/issues/new).
- Add the relevant tags.
- Explain the behavior you would expect and the actual behavior.
- Please provide as much context as possible and describe the *reproduction steps* that someone else can follow to recreate the issue on their own. This usually includes your code. For good bug reports you should isolate the problem and create a reduced test case.
- Provide the information you collected in the previous section.

Once it's filed:

- A maintainer will try to reproduce the issue with your provided steps. If there are no reproduction steps or no obvious way to reproduce the issue, a maintainer will ask you for those steps and mark the issue as `needs-repro`. Bugs with the `needs-repro` tag will not be addressed until they are reproduced.
- If a maintainer is able to reproduce the issue, it will be marked `needs fix`, as well as possibly other tags (such as `critical`), and the issue will be worked on by a maintainer or a contributor.

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for Synapsea, **including completely new features and minor improvements to existing functionality**. Following these guidelines will help maintainers and the community to understand your suggestion and find related suggestions.

#### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the [documentation](https://sites.google.com/view/synapsea/api) carefully and find out if the functionality is already covered, maybe by an individual configuration.
- Perform a [search](https://github.com/mochji/synapsea/issues) to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.
- Find out whether your idea fits with the scope and aims of the project. It's up to you to make a strong case to convince the project's developers of the merits of this feature. Keep in mind that we want features that will be useful to the majority of our users and not just a small subset. If you're just targeting a minority of users, consider writing an add-on/plugin library.

#### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues](https://github.com/mochji/synapsea/issues).

- Use a **clear and descriptive title** for the issue to identify the suggestion.
- Provide a **step-by-step description of the suggested enhancement** in as many details as possible.
- **Describe the current behavior** and **explain which behavior you expected to see instead** and why. At this point you can also tell which alternatives do not work for you.
- You may want to **include screenshots and animated GIFs** which help you demonstrate the steps or point out the part which the suggestion is related to.
- **Explain why this enhancement would be useful** to most Synapsea users. You may also want to point out the other projects that solved it better and which could serve as inspiration.

## Coding styleguide

### Style

#### Indentation

 - Use tabs for each indentation level.

#### Naming

 - Use descriptive and meaningful names for variables and functions.
 - Use `camelCase` for names.
 - Constants should be in `CAPITALS_WITH_UNDERSCORES`.

#### Errors

 - Errors should follow the style of Lua errors.

#### Comments

 - Use comments sparingly.
 - Include a single space after the initial `--`.
 - Use single-line comments `--` for single-line comments and block comments `--[[ ... ]]--` for block comments.

#### Tokens

 - Include parenthesis when calling functions.
 - Use double quotes `"`.

#### Whitespace

 - Include a whitespace after mathematical tokens.
 - Do not have whitespace before commas `,`.
 - Do not have trailing whitespace at the end of lines.
 - **Do not** use DOS/Windows newlines (LF `\n` instead of CRLF `\r\n`).
 - Blank lines should separate logical sections of the code.

### Logic

#### Functions

 - Functions should do 1 - 2 concepts.
 - Functions should not be absurdly wrong.
 - Only abstract after 3 repeats to avoid coupling.

#### Comments

 - Use comments sparingly.
 - Only use comments to explain *why* it's like that, not *what* it does.
 - You shouldn't have to use comments to explain your code.

#### Variables

 - Declare variables at the lowest scope possible.
 - **Do not** use global variables.

#### Errors

 - Use specific exceptions.
 - Provide information about the error.
 - Handle errors at an appropriate place in the code.

### Best practices

 - Avoid magic numbers.
 - Write modular and expandable code.
 - Periodically review and refactor code for readability and maintainability to avoid software rot.

## Attribution
This guide is based on the [**contributing-gen**](https://github.com/bttger/contributing-gen).
