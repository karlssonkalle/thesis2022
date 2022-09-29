#Altmetric.com asks users not to hammer the server. If needed, add a delay to the API-calls. 
#Packages:
from pyaltmetric import Altmetric, Citation, HTTPException
import csv

# initialize Altmetric
a = Altmetric()

with open('Altmetric_dois.csv', 'r') as infile:
    with open('Altmetric_result.csv', 'w') as outfile:
        writer = csv.writer(outfile)

        for x in infile:
            x = x.rstrip()
            # search for article using doi
            c = a.doi(x)
            if c:
                # initialize Citation and fetch fields
                citation = Citation(c)
                result = citation.get_fields('doi','title','cited_by_accounts_count', 'cited_by_posts_count', 'cited_by_msm_count', 'cited_by_policies_count', 'cited_by_tweeters') #Choose which data to retrieve. 
            else:
                result = [x, 'No data'] #No data = DOI not found. 
            # write row to file
            writer.writerow(result)