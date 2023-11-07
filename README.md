# ChatPI

Authors: Andrei Cozma, Manan Patel, Tulsi Tailor, Zac Perry

Objective: Develop a multi-lingual mini chatbot (ChatPI) that can take natural language queries about the plot of novels, answer questions in two different languages, and summarize the key points about the plot sections.

## Deliverables

- Notebook (make sure it runs on Google Colab)
- 7-10 slides
- up to 4-page report, excluding references

### Project Description

#### Part 1 - Question-Answering pipeline

**_Objective_** - Implement a prompt interface that takes in a question, runs it through the question-answering pipeline and returns the answer
**_Tasks_**:

- Find five 300-words sections from a book that introduces the following:

  - Protagonist

  - Antagonist

  - Crime and crime scene

  - Any significant evidence

  - Resolution of crime/a narrative that presents the case against perpetrator

- Ask the model questions and return the answers
- Document the results
- Use different Question-Answering model to do same tasks mentioned above and document differences in the results

---

#### Part 2 - Translation pipeline (French)

**_Objective_** - Utilize a translation pipeline that translates the answers found in Part 1 into French and back to English
**_Tasks_**:

```md
> Question
> Answer in English
> Answer in French
> Answer in English, translated from French
```

- Document the results
- Use different Translation model to do same tasks mentioned above and document differences in the results

---

#### Part 3 - Summarization pipeline

**_Objective_** - Utilize a text summarization pipeline that summarizes the 300-words sections found in Part 1
**_Tasks_**:

- Run the five 300-words sections through pipeline
- Document the results
- Use different text summarization model to do same tasks mentioned above and document differences in the results
