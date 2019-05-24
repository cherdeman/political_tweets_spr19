from utils.db_client import DBClient
import pandas as pd

if __name__ == "__main__":   
    COUNTS_BY_TOPICS = "select sum(budget::int) as budget, sum(civil_rights::int) as civil_rights, sum(courts::int) as courts, sum(criminal_justice::int) as criminal_justice, sum(drugs::int) as drugs, sum(econ_inequality::int) as econ_inequality, sum(econ_jobs::int) as econ_jobs, sum(education::int) as education, sum(environment::int) as environment, sum(family::int) as family, sum(foreign_policy::int) as foreign_policy, sum(governance::int) as governance, sum(guns::int) as guns, sum(health::int) as health,  sum(immigration::int) as immigration, sum(military::int) as military,  sum(public_safety::int) as public_safety, sum(puerto_rico::int) as puerto_rico, sum(race::int) as race, sum(rural::int) as rural, sum(russia::int) as russia, sum(sexual_assault::int) as sexual_assault, sum(shutdown::int) as shutdown,  sum(social_security::int) as social_security, sum(taxes::int) as taxes, sum(technology::int) as technology, sum(women_rights::int) as women_rights from staging.master"
    db = DBClient()
    with db.conn.cursor() as cur:
        cur.execute(COUNTS_BY_TOPICS)
        colnames = [desc[0] for desc in cur.description]
        print(colnames)
        results = cur.fetchall()
    table = pd.DataFrame(results, columns = colnames)
    #table_t = table.transpose()
    table.to_csv("count_by_topic.csv", index=False)