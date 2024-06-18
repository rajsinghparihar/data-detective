# Introduction 
TODO: Give a short introduction of your project. Let this section explain the objectives or the motivation behind this project. 

# Getting Started
TODO: Guide users through getting your code up and running on their own system. In this section you can talk about:
1.	Installation process
2.	Software dependencies
3.	Latest releases
4.	API references

# Build and Test
TODO: Describe and show how to build your code and run the tests. 

# Contribute
TODO: Explain how other users and developers can contribute to make your code better. 

If you want to learn more about creating good readme files then refer the following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore)


```json
{
    process_id: { 
        "$in": [
            "0466f322-27e4-11ef-8f70-1ec97df178de", 
            "491d1f8a-2895-11ef-8c52-1ec97df178dd", 
            "91f64424-28a0-11ef-87d1-1ec97df178dd", 
            "609436a8-28a4-11ef-87d1-1ec97df178dd",
            "61e0e2e0-29a8-11ef-97d4-1ec97df178dd",
            "94d1ba02-29bd-11ef-ae52-1ec97df178dd",
            "68b62a3c-29bf-11ef-ae52-1ec97df178dd",
            "65ea13b0-29c2-11ef-a483-1ec97df178dd"
        ] 
    }
}
```
```go
"0466f322-27e4-11ef-8f70-1ec97df178de", // readable files < 10kb 66 files (65/66 processed, 1 remaining)
"491d1f8a-2895-11ef-8c52-1ec97df178dd", // readable files first batch 19 files (15/19 processed, 4 remaining)
"91f64424-28a0-11ef-87d1-1ec97df178dd", // readable files first batch second iteration (3 / remaining 4 processed, 1 remaining)
"609436a8-28a4-11ef-87d1-1ec97df178dd", // ocr files first batch first iteration (20 / 44 processed, 24 remaining)
"61e0e2e0-29a8-11ef-97d4-1ec97df178dd", // ocr files first batch second iteration (22 / remaining 24 processed, 2 remaining)
"ce1aade8-29ac-11ef-97d4-1ec97df178dd", // ocr files second batch first iteration (5 / 10 processed, 5 remaining)
"94d1ba02-29bd-11ef-ae52-1ec97df178dd", // ocr files second batch second iteration (5 / 5 remaining processed. 
"68b62a3c-29bf-11ef-ae52-1ec97df178dd" // terrible results all files processed but have a lot of missing values
"65ea13b0-29c2-11ef-a483-1ec97df178dd" // terrible pdfs second iteration.

```

show entities for selected document type
update document type if it exists