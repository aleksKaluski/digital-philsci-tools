from pymongo import MongoClient, InsertOne
from colorama import Fore, Style, init
import time

MONGO_IP = "localhost"
MONGO_PORT = 27017


def main():
    client = MongoClient(MONGO_IP, MONGO_PORT)

    if client is not None:
        db = client.papers_db
        collection = db.papers

        test_papers = [
            {
                "corpusid": 1,
                "title": "Verificationism Then and Now",
                "abstract": "The term verificationism is used in two different ways: the first is in relation to the verification principle of meaning, which we usually and rightly associate with the logical empiricists, although, as we now know, it derives in reality from Wittgenstein, and the second is in relation to the theory of meaning for intuitionistic logic that has been developed, beginning of course with Brouwer, Heyting and Kolmogorov in the 1920s and early 1930s but in much more detail lately, particularly in connection with intuitionistic type theory. It is therefore very natural to ask how these two forms of verificationism are related to one another: was the verificationism that we had in the 1930s a kind of forerunner of what we have now, or was it something entirely different? I would like to discuss this question by considering a very particular problem, which was at the heart of Schlick’s interests, namely, the problem whether there might exist undecidable propositions or, if you prefer, unsolvable problems or unanswerable questions: it is merely a matter of wording which of these terms you choose. As I said, it is a problem which was at the heart of Schlick’s interests: it is explicitly discussed already in his early, programmatic paper Die Wende in der Philosophie in the first volume of Erkenntnis from 1930, and there is a short later paper, which has precisely Unanswerable Questions? as its title, from 1935, and he discussed it on several occasions in between also.",
                "authors": [{"name": "Per Martin-Lof"}]
            },
            {
                "corpusid": 2,
                "title": "Language, Truth, and Logic",
                "abstract": "Language, Truth and Logic is a 1936 book about meaning by the philosopher Alfred Jules Ayer, in which the author defines, explains, and argues for the verification principle of logical positivism, sometimes referred to as the criterion of significance or criterion of meaning. Ayer explains how the principle of verifiability may be applied to the problems of philosophy. Language, Truth and Logic brought some of the ideas of the Vienna Circle and the logical empiricists to the attention of the English-speaking world.",
                "authors": [{"name": "Alfred Jules Ayer"}]
            },
            {
                "corpusid": 3,
                "title": "Meaning and Verification",
                "abstract": "The verifiability theory of meaning says that meaning is evidence. It is anticipated in, for example, Hume’s empiricist doctrine of impressions and ideas, but it emerges into full notoriety in twentieth-century logical positivism. The positivists used the theory in a critique of metaphysics to show that the problems of philosophy, such as the problem of the external world and the problem of other minds, are not real problems at all but only pseudoproblems. Their publicists used the doctrine to argue that religion, ethics and fiction are meaningless, which is how verificationism became notorious among the general public.",
                "authors": [{"name": "Moritz Schlick"}]
            }
        ]

        start = time.time()
        print('\n' + Fore.BLUE + Style.BRIGHT + '=' * 60)
        print(Fore.YELLOW + Style.BRIGHT + f"{'Tesing MongoDB':^60}")
        print(Fore.BLUE + Style.BRIGHT + '-' * 60)

        # Bulk insert
        operations = [InsertOne(paper) for paper in test_papers]
        result = collection.bulk_write(operations)
        print(Fore.GREEN + "[+] " + Fore.RESET + f"Inserted {result.inserted_count} papers.")

        # create index
        collection.create_index("corpusid")
        print(Fore.GREEN + "[+] " + Fore.RESET + f"Created corpusid index")

        # test query
        found = collection.find_one({"corpusid": 1})
        print(Fore.GREEN + "[+] " + Fore.RESET + f"Found paper: {found['title']}")

        # count
        count = collection.count_documents({})
        print(Fore.GREEN + "[+] " + Fore.RESET + f"Total papers in collection: {count}")

        client.close()
        end = time.time()
        print(Fore.GREEN + Style.BRIGHT + f"[SUCCESS] Total runtime: {end - start:.2f} seconds" + Style.RESET_ALL)
        print(Fore.BLUE + Style.BRIGHT + '=' * 60 + Style.RESET_ALL)


if __name__ == '__main__':
    main()