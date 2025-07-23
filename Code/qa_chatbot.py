from medicine_retriever import MedicineRetriever

def main():
    retriever = MedicineRetriever()
    print("Medicine retrieval chatbot ready! Type 'exit' to quit.")
    while True:
        q = input("You: ")
        if q.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        answers = retriever.query(q)
        if not answers:
            print("No relevant information found.")
        else:
            for i, a in enumerate(answers, 1):
                print(f"\nResult {i}:")
                print(f"Name: {a['Medicine Name']}")
                print(f"Composition: {a['Composition']}")
                print(f"Uses: {a['Uses']}")
                print(f"Side Effects: {a['Side Effects']}")
                print(f"Manufacturer: {a['Manufacturer']}")
                print(f"Reviews - Excellent: {a['Excellent Review %']}%, "
                      f"Average: {a['Average Review %']}%, "
                      f"Poor: {a['Poor Review %']}%\n")

if __name__ == "__main__":
    main()
