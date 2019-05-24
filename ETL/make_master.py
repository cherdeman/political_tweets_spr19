from utils.db_client import DBClient

master_query = """
drop table if exists staging.master cascade;
create table staging.master as (
	select *, true as democrat 
	from staging.democrat
	union
	select *, false as democrat 
	from staging.republican
	union
	select h.*, case when ha.party_affiliation = 'D' then true
					 when ha.party_affiliation = 'R' then false
					 else null end as democrat
	from staging.house h
	join raw.house_accounts ha
	using(user_id)
	union
	select s.*, case when sa.party_affiliation = 'D' then true
					 when sa.party_affiliation = 'R' then false
					 else null end as democrat
	from staging.senate s
	join raw.senate_accounts sa
	using(user_id)
)
"""


if __name__ == "__main__":
	db = DBClient()
	print("Attempting to make the master table...")
	db.write([master_query])
	print("Master table done!")
	db.exit()
	print("Bye.")