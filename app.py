import time
import os
import numpy as np
import streamlit as st
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine
from datetime import datetime
from datetime import date
from dotenv import load_dotenv
from urllib.parse import quote_plus

# Retrieve contents from .env file
load_dotenv()
DB_IP = os.getenv('DB_IP')
DB_PORT = os.getenv('DB_PORT')
DB_USER = os.getenv('DB_USER')
DB_PASS = quote_plus(os.getenv('DB_PASS'))
DB = os.getenv('DB')

# Function to create SQLAlchemy engine
def get_db_connection():
    engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_IP}:{DB_PORT}/{DB}')
    return engine.connect()

# Cache the function that fetches data from the SQL table
@st.cache_data(ttl='1d')
def fetch_data_from_sql(query):
    conn = get_db_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Function to update SQL table
def update_sql_table(data, table_name, schema, replace=True):
    conn = get_db_connection()
    data.to_sql(table_name, conn, if_exists='replace' if replace else 'append', index=False, schema=schema)
    conn.close()

# Get Prospects
def get_prospects(update=False):
    if update:
        try:
            query = '''
                            WITH
                                prospects AS (
                                    SELECT
                                        lookup_id AS constituent_id,
                                        id AS system_record_id
                                    FROM
                                        constituent_list AS cl
                                        INNER JOIN donor_research.prospects AS ab ON ab."RE ID" = cl.lookup_id
                                )

                            SELECT
                                CONCAT(name, ' (', id, ')') AS prospects
                            FROM
                                prospects AS ps
                                LEFT JOIN constituent_list AS cl ON cl.id = ps.system_record_id
                                WHERE
                                    id IN (
                                            SELECT
                                                parent_id
                                            FROM
                                                donor_research.employment
                                        );
                            '''

            data = fetch_data_from_sql(query)

        except SQLAlchemyError:
            st.warning('No prospects researched yet!')
            st.stop()

    else:
        try:
            query = '''
                    WITH
                        prospects AS (
                            SELECT
                                lookup_id AS constituent_id,
                                id AS system_record_id
                            FROM
                                constituent_list AS cl
                                INNER JOIN donor_research.prospects AS ab ON ab."RE ID" = cl.lookup_id
                            WHERE
                                id NOT IN (
                                    SELECT
                                        parent_id
                                    FROM
                                        donor_research.employment
                                    )    
                        )
        
                    SELECT
                        CONCAT(name, ' (', id, ')') AS prospects
                    FROM
                        prospects AS ps
                            LEFT JOIN constituent_list AS cl ON cl.id = ps.system_record_id;
                    '''

            data = fetch_data_from_sql(query)

        except SQLAlchemyError:
            query = '''
                    WITH
                        prospects AS (
                            SELECT
                                lookup_id AS constituent_id,
                                id AS system_record_id
                            FROM
                                constituent_list AS cl
                                INNER JOIN donor_research.prospects AS ab ON ab."RE ID" = cl.lookup_id
                        )
    
                    SELECT
                        CONCAT(name, ' (', id, ')') AS prospects
                    FROM
                        prospects AS ps
                            LEFT JOIN constituent_list AS cl ON cl.id = ps.system_record_id;
                    '''

            data = fetch_data_from_sql(query)

    return data['prospects'].to_list()

def get_basic_from_re(re_id):
    query = f'''
                SELECT
                    lookup_id AS constituent_id,
                    name,
                    age,
                    gender,
                    address_city AS city,
                    TRIM(CONCAT(address_county, ' ', address_state)) AS state,
                    address_country AS country,
                    deceased,
                    cl.inactive,
                    CASE
                        WHEN LEFT(ol.address, 4) = 'http' THEN ol.address 
                        ELSE CONCAT('https://', ol.address)
                    END AS linkedin,
                    CONCAT('https://host.nxt.blackbaud.com/constituent/records/',
                           cl.id,'?envId=p-dzY8gGigKUidokeljxaQiA&svcId=renxt') AS re_profile_link
                FROM
                    constituent_list cl
                    LEFT JOIN online_presence_list ol ON ol.constituent_id = cl.id
                WHERE
                    cl.id = '{re_id}' AND
                    ol.type LIKE 'LinkedIn%%'
                LIMIT 1;
            '''

    data = fetch_data_from_sql(query)

    return data

def get_donations(re_id):
    query = f'''
                SELECT
                    CONCAT(
                            RIGHT(REPLACE(date, 'T00:00:00', ''), 2),
                            '-',
                            SUBSTRING(date, 6, 2),
                            '-',
                            LEFT(date, 4)
                    ) AS date,
                    cl.description AS project,
                    CASE
                        WHEN gift_splits_0_fund_id = '457' THEN 'Heritage Fund'
                        WHEN gift_splits_0_fund_id = '457' THEN 'IITAAUK'
                        WHEN gift_splits_0_fund_id = '469' THEN 'Alumni Association'
                        WHEN gift_splits_0_fund_id = '470' THEN 'HF Funds Raised but not received'
                        WHEN gift_splits_0_fund_id = '471' THEN 'HF - Not Received'
                        ELSE 'Alumni & Corporate Relationship'
                    END AS office,
                    amount_value AS amount
                FROM
                    gift_list gl
                    LEFT JOIN campaign_list cl ON cl.id = CAST(gl.gift_splits_0_campaign_id AS INT)
                WHERE
                    gl.constituent_id = '{re_id}' AND
                    amount_value > 0 AND
                    type IN ('Donation', 'GiftInKind')
                ORDER BY
                    gl.date DESC;
            '''

    data = fetch_data_from_sql(query)

    data_1 = pd.DataFrame(data={
            'Lifetime Donation': data['amount'].sum(),
        }, index=[0])

    return data, data_1

def get_employment(re_id):
    query = f'''
                SELECT
                    name AS organisation,
                    position,
                    CAST(start_y AS TEXT) AS start_year,
                    CAST(end_y AS TEXT) AS end_year
                FROM
                    relationship_list
                WHERE
                    (
                        type = 'Employee' OR
                        type = 'Employer' OR
                        reciprocal_type = 'Employee' OR
                        reciprocal_type = 'Employer'
                    ) AND
                    constituent_id = '{re_id}'
                ORDER BY
                    end_y DESC, start_y DESC;
            '''

    data = fetch_data_from_sql(query)

    return data

def get_education(re_id):
    query = f'''
                SELECT
                    CASE 
                        WHEN school = 'Indian Institute of Technology Bombay' THEN 'IIT Bombay'
                        ELSE school 
                    END AS school,
                    class_of,
                    degree,
                    majors_0 AS department,
                    social_organization AS hostel
                FROM
                    school_list
                WHERE
                    constituent_id = '{re_id}'
                UNION
                SELECT
                    name AS school,
                    CAST(end_y AS TEXT) AS class_of,
                    position AS degree,
                    NULL AS department,
                    NULL AS hostel
                FROM
                    relationship_list
                WHERE
                    (
                        type = 'Student' OR
                        reciprocal_type = 'Student'
                    ) AND
                    constituent_id = '{re_id}'
                ORDER BY
                    class_of DESC;
            '''

    data = fetch_data_from_sql(query)

    return data

def get_awards(re_id):
    query = f'''
                SELECT
                    value AS award,
                    CONCAT(
                        RIGHT(REPLACE(date, 'T00:00:00', ''), 2),
                        '-',
                        SUBSTRING(date, 6, 2),
                        '-',
                        LEFT(date, 4)
                    ) AS date
                FROM
                    constituent_custom_fields
                WHERE
                    category = 'Awards' AND
                    parent_id = '{re_id}';
            '''

    data = fetch_data_from_sql(query)

    return data

def process_live_alumni(data):
    # Load to dataframe
    data = pd.read_csv(data)

    # Remove incorrect values
    data['Employment Salary Min'] = data['Employment Salary Min'].replace('="0"', np.nan)
    data['Employment Salary Max'] = data['Employment Salary Max'].replace('="0"', np.nan)
    data['Person Rating (stars)'] = data['Person Rating (stars)'].replace('="0"', np.nan)

    # Converting date-time column
    data['Employment Captured Date'] = pd.to_datetime(data['Employment Captured Date'], format='%m/%d/%Y %H:%M:%S %p')

    # Drop total column
    data = data.drop(columns=['_totalcount_'])

    # Add Live Alumni URL
    data['Live Alumni URL'] = 'https://app.livealumni.com/people/details/' + data['id'].astype(str)

    update_sql_table(data, 'live_alumni', 'donor_research')

# Function to format amount in Indian number format (lakhs, crores)
def format_inr(amount):
    # Convert the number to a string
    num_str = str(amount)

    # Handle decimal part if any
    if '.' in num_str:
        int_part, dec_part = num_str.split('.')
    else:
        int_part, dec_part = num_str, None

    # Start from the right side and add commas after every 2 digits, except for the first group (3 digits)
    last_three = int_part[-3:]
    remaining = int_part[:-3]

    if remaining != '':
        remaining = ','.join([remaining[max(i - 2, 0):i] for i in range(len(remaining), 0, -2)][::-1])
        formatted_number = f'{remaining},{last_three}'
    else:
        formatted_number = last_three

    return f'₹ {formatted_number}'

def display_re_data(selection):
    name = ' '.join(selection.split('(')[0].split()[:-1]) if selection.split('(')[0].split()[-1].isnumeric() \
        else selection.split('(')[0]
    system_record_id = selection.split('(')[1].strip(')')

    st.title(name)
    st.divider()

    # Get Basic info of Prospect from RE
    re_basic = get_basic_from_re(system_record_id)

    st.subheader('Data from Raisers Edge')
    st.text('')
    st.dataframe(
        re_basic.drop(columns=['name']),
        use_container_width=True,
        hide_index=True,
        column_config={
            'constituent_id': st.column_config.NumberColumn(
                'RE ID',
                format = '%d'
            ),
            'age': 'Age',
            'gender': 'Gender',
            'city': 'City',
            'state': 'State',
            'country': 'Country',
            'deceased': 'Deceased',
            'inactive': 'Inactive',
            'linkedin': st.column_config.LinkColumn(
                'LinkedIn',
                display_text='Open LinkedIn',
                width='small'
            ),
            're_profile_link': st.column_config.LinkColumn(
                'Raisers Edge Profile',
                display_text='Open in RE',
                width='small'
            )
        }
    )

    col1, col2 = st.columns(2)

    with col1:
        # Employment from RE
        st.write('**Employment**')
        re_employment = get_employment(system_record_id)
        st.dataframe(
            re_employment,
            hide_index=True,
            use_container_width=True,
            column_config={
                'organisation': 'Organisation',
                'position': 'Position',
                'start_year': st.column_config.NumberColumn(
                    'Joining Year',
                    format='%d'
                ),
                'end_year': st.column_config.NumberColumn(
                    'End Year',
                    format='%d'
                )
            }
        )

        # Education from RE
        st.write('**Education**')
        re_education = get_education(system_record_id)
        st.dataframe(
            re_education,
            hide_index=True,
            use_container_width=True,
            column_config={
                'school': 'School',
                'class_of': st.column_config.NumberColumn(
                    'Batch',
                    format='%d'
                ),
                'degree': 'Degree',
                'department': 'Department',
                'hostel': 'Hostel'
            }
        )

    with col2:
        re_donations, re_lifetime_donations = get_donations(system_record_id)
        re_awards = get_awards(system_record_id)

        # Assuming re_donations is your DataFrame
        re_lifetime_donations['Lifetime Donation'] = re_lifetime_donations['Lifetime Donation'].apply(format_inr)

        # Check if there are any donations
        if re_donations.empty:
            col3, col4 = st.columns(2)

            with col3:
                st.write(f'**Donations**')

                st.dataframe(
                    re_lifetime_donations,
                    hide_index=True,
                    use_container_width=True,
                )

            with col4:
                st.write('**Awards**')

                re_awards['date'] = pd.to_datetime(re_awards['date'])

                st.dataframe(
                    re_awards,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        'award': 'Awards',
                        'date': st.column_config.DateColumn(
                        'Date',
                        format='D MMM YYYY'
                    )
                    }
                )

        else:
            st.write('**Donations**')

            re_donations['date'] = pd.to_datetime(re_donations['date'], dayfirst=True)

            # Assuming re_donations is your DataFrame
            re_donations['amount'] = re_donations['amount'].apply(format_inr)

            st.dataframe(
                re_donations,
                hide_index=True,
                use_container_width=True,
                column_config={
                    'date': st.column_config.DateColumn(
                        'Date',
                        format='D MMM YYYY'
                    ),
                    'project': 'Project / Campaign',
                    'office': 'Office',
                    'amount': st.column_config.TextColumn(
                        'Amount'
                    )
                }
            )

            col3, col4 = st.columns(2)

            with col3:
                st.write('**Awards**')

                re_awards['date'] = pd.to_datetime(re_awards['date'], dayfirst=True)

                st.dataframe(
                    re_awards,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        'award': 'Awards',
                        'date': st.column_config.DateColumn(
                            'Date',
                            format='D MMM YYYY'
                        )
                    }
                )

            with col4:
                st.write('**Donation Summary**')
                st.dataframe(
                    re_lifetime_donations,
                    hide_index=True,
                    use_container_width=True
                )

    return system_record_id

def usd_number_format(number):

    if number is None:
        return number

    else:

        # Convert the number to a string
        num_str = str(number)

        # Handle decimal part if any
        if '.' in num_str:
            int_part, dec_part = num_str.split('.')
        else:
            int_part, dec_part = num_str, None

        # Start from the right side and add commas after every 3 digits
        int_part_with_commas = ','.join([int_part[max(i - 3, 0):i] for i in range(len(int_part), 0, -3)][::-1])

        # Add decimal part back if there was one
        if dec_part:
            formatted_number = f'{int_part_with_commas}.{dec_part}'
        else:
            formatted_number = int_part_with_commas

        return f'$ {formatted_number}'

def display_live_alumni_data(system_record_id,):
    st.subheader('')
    st.subheader('Data from Live Alumni')

    data = get_live_alumni(system_record_id)

    if data.empty:
        st.write('No data in Live Alumni')

    else:
        # Employment Data
        st.write('')
        col5, col6 = st.columns(2)

        with col5:
            st.write('**Employment**')
            st.dataframe(
                data[[
                    'Employment Company Name',
                    'Employment Title',
                    'Employment Start Year',
                    'Employment End Year'
                ]].drop_duplicates(),
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Employment Company Name': 'Organisation',
                    'Employment Title': 'Position',
                    'Employment Start Year': st.column_config.NumberColumn(
                        'Joining Year',
                        format='%d'
                    ),
                    'Employment End Year': st.column_config.NumberColumn(
                        'End Year',
                        format='%d'
                    )
                }
            )

            st.write('**Position Details**')

            data['Employment Salary Min'] = data['Employment Salary Min'].apply(usd_number_format)
            data['Employment Salary Max'] = data['Employment Salary Max'].apply(usd_number_format)

            st.dataframe(
                data[[
                    'Employment Title Is Senior',
                    'Employment Seniority Level',
                    'Employment Salary Min',
                    'Employment Salary Max',
                    'Person Rating (stars)'
                ]].drop_duplicates(),
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Employment Title Is Senior': 'Is Senior?',
                    'Employment Seniority Level': 'Seniority Level',
                    'Employment Salary Min': 'Min. Salary',
                    'Employment Salary Max': 'Max. Salary'
                }
            )

        with col6:
            col7, col8 = st.columns([0.8, 0.2])

            with col7:
                st.write('**Headline**')
                st.dataframe(
                    data[[
                        'Person Headline'
                    ]].drop_duplicates(),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Person Headline': 'Headline on LinkedIn'
                    }
                )

            with col8:
                st.write('**Online Links**')
                st.dataframe(
                    data[[
                        'Live Alumni URL',
                        'Person URL'
                    ]].drop_duplicates(),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Live Alumni URL': st.column_config.LinkColumn(
                            'Live Alumni',
                            display_text='Open',
                            width='small'
                        ),
                        'Person URL': st.column_config.LinkColumn(
                            'LinkedIn',
                            display_text='Open',
                            width='small'
                        )
                    }
                )

            col9, col10 = st.columns([0.8, 0.2])

            with col9:
                st.write('**Location**')
                loc = [col for col in data.columns if 'Location' in col]
                loc_1 = [col.split()[1] for col in data.columns if 'Location' in col]

                st.dataframe(
                    data[
                        loc
                    ].drop_duplicates(),
                    use_container_width=True,
                    hide_index=True,
                    column_config=dict(zip(loc, loc_1))
                )

            with col10:
                st.write('**Rating**')
                rating = data['Person Rating (stars)'][0]
                if rating is None:
                    st.caption('No rating')

                else:
                    st.markdown('⭐️' * int(data['Person Rating (stars)'][0]))

        col11, col12 = st.columns(2)

        with col11:
            st.write('**Industry Details**')
            st.dataframe(
                data[
                    [col for col in data.columns if col.startswith('Company')]
                ].drop_duplicates(),
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Company Industry Name': 'Industry',
                    'Company Type Type': 'Type',
                    'Company Details Size': 'Size',
                    'Company Details Sector': 'Sector'
                }
            )

        with col12:
            st.write('**Education**')
            st.dataframe(
                data[[
                    'University Name',
                    'Education Start Date',
                    'Education End Date',
                    'Education Degree',
                    'Education Major'
                ]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    'University Name': 'University',
                    'Education Start Date': st.column_config.NumberColumn(
                        'Joining Year',
                        format='%d',
                        width='small'
                    ),
                    'Education End Date': st.column_config.NumberColumn(
                        'Graduated on',
                        format='%d',
                        width='small'
                    ),
                    'Education Degree': 'Degree',
                    'Education Major': 'Department'
                }
            )

def get_live_alumni(re_id):
    query = f'''
                WITH
                    la_re_mapping AS (
                        SELECT
                            parent_id AS re_id,
                            CAST(value AS INT) AS live_alumni_id
                        FROM
                            constituent_custom_fields
                        WHERE
                            category = 'Live Alumni ID'
                    )
                
                SELECT
                    *
                FROM
                    donor_research.live_alumni AS la
                    RIGHT JOIN la_re_mapping AS map ON map.live_alumni_id = la.id
                WHERE
                    map.re_id = '{re_id}'
            '''

    data = fetch_data_from_sql(query)

    return data

def save_manual_employment(re_id):
    st.write('**Employment** `(auto-populated from Live Alumni & Raisers Edge)`')
    re_employment = get_employment(re_id)

    # RE Data
    re_employment = re_employment.rename(columns={
        'organisation': 'Organisation',
        'position': 'Position',
        'start_year': 'Joining Year',
        'end_year': 'End Year'
    })

    # Live Alumni Data
    la_employment = get_live_alumni(re_id)

    la_employment = la_employment.drop_duplicates()

    la_employment = la_employment[['Employment Company Name', 'Employment Title', 'Employment Start Year',
                                   'Employment End Year']]

    la_employment = la_employment.rename(columns={
        'Employment Company Name': 'Organisation',
        'Employment Title': 'Position',
        'Employment Start Year': 'Joining Year',
        'Employment End Year': 'End Year'
    })

    data = pd.concat([la_employment, re_employment], ignore_index=True, axis=0)

    data['Organisation'] = data['Organisation'].str.strip()
    data['Position'] = data['Position'].str.strip()

    data['Industry'] = None
    data['Is Senior?'] = None
    data['Salary'] = None
    data['Stakes left'] = None
    data['Shares owned'] = None
    data['Funding Stage'] = None
    data['Company Valuation'] = None
    data['Share Price'] = None
    data['Comments'] = None

    data = data.drop_duplicates(ignore_index=True).reset_index(drop=True)

    return data

def save_manual_education(re_id):

    # RE Data
    re_education = get_education(re_id)

    re_education = re_education.rename(columns={
        'school': 'University',
        'hostel': 'Hostel',
        'class_of': 'Graduated on',
        'degree': 'Degree',
        'department': 'Department'
    })

    # Live Alumni
    la_education = get_live_alumni(re_id)

    la_education = la_education[[
                    'University Name',
                    'Education Start Date',
                    'Education End Date',
                    'Education Degree',
                    'Education Major'
                ]]

    la_education = la_education.rename(columns={
        'University Name': 'University',
        'Education Start Date': 'Joining Year',
        'Education End Date': 'Graduated on',
        'Education Degree': 'Degree',
        'Education Major': 'Department'
    })

    data = pd.concat([re_education, la_education], ignore_index=True, axis=0)

    # data = data[['University', 'Joining Year', 'Graduated on', 'Degree', 'Department', 'Hostel']]
    data = data[['University', 'Joining Year', 'Graduated on', 'Degree', 'Department']]

    data = data.drop_duplicates(ignore_index=True).reset_index(drop=True)

    data = data.sort_values(by=['Graduated on', 'Joining Year'], ascending=[False, True], ignore_index=True)

    return data

def save_manual_location():
    data = pd.DataFrame(data={
        'Location': [None]
    })

    return data

def save_manual_net_worth():
    data = pd.DataFrame(data={
        'Net worth': [np.nan]
    })

    return data

def save_manual_philanthropy():
    data = pd.DataFrame(data={
        'Foundation': [None],
        'Date': [None],
        'Amount': [None],
        'Comments': [None]
    })

    return data

def get_df_shape_product(df):
    return df.shape[0] * df.shape[1]

def get_null_values(df):
    return df.replace('', None).isnull().sum().sum()

def get_research(re_id):
    st.write('**Employment**')
    employment = get_research_employment(re_id)

    col1, col2 = st.columns(2)

    with col1:
        st.write('**Education**')
        education = get_research_education(re_id)

    with col2:
        st.write('**Philanthropy**')
        philanthropy = get_research_philanthropy(re_id)

        col3, col4 = st.columns([0.6, 0.4])

        with col3:
            st.write('**Location**')
            location = get_research_location(re_id)

        with col4:
            st.write('**Net Worth**')
            net_worth = get_research_net_worth(re_id)

    st.write('**Remarks**')
    remarks = get_research_remarks(re_id)

    return employment, education, philanthropy, location, net_worth, remarks

def get_research_employment(re_id):
    query = f'''
            SELECT
                "Organisation",
                "Position",
                "Joining Year",
                "End Year",
                "Industry",
                "Is Senior?",
                "Salary",
                "Stakes left",
                "Shares owned",
                "Funding Stage",
                "Company Valuation",
                "Share Price",
                "Comments"
            FROM
                donor_research.employment
            WHERE
                parent_id = '{re_id}' AND
                updated_on = (
                    SELECT
                        updated_on
                    FROM
                        donor_research.employment
                    WHERE
                        parent_id = '{re_id}'
                    ORDER BY
                        updated_on DESC
                    LIMIT 1
                );
            '''

    try:
        data = fetch_data_from_sql(query)

    except SQLAlchemyError:
        data = pd.DataFrame(data={
            'Organisation': [None],
            'Position': [None],
            'Joining Year': [None],
            'End Year': [None],
            'Industry': [None],
            'Is Senior?': [None],
            'Salary': [np.nan],
            'Stakes left': [np.nan],
            'Shares owned': [np.nan],
            'Funding Stage': [None],
            'Company Valuation': [np.nan],
            'Share Price': [np.nan],
            'Comments': [None]
        })

    data = st.data_editor(
        data,
        num_rows='dynamic',
        use_container_width=True,
        hide_index=True,
        column_config=employment_config()
    )

    return data

def get_research_education(re_id):
    query = f'''
            SELECT
                "University",
                "Joining Year",
                "Graduated on",
                "Degree",
                "Department"
            FROM
                donor_research.education
            WHERE
                parent_id = '{re_id}' AND
                updated_on = (
                    SELECT
                        updated_on
                    FROM
                        donor_research.education
                    WHERE
                        parent_id = '{re_id}'
                    ORDER BY
                        updated_on DESC
                    LIMIT 1
                );
            '''

    try:
        data = fetch_data_from_sql(query)

    except SQLAlchemyError:
        data = pd.DataFrame(data={
            'University': [None],
            'Joining Year': [None],
            'Graduated on': [None],
            'Degree': [None],
            'Department': [None],
            'Hostel': [None]
        })

    data = st.data_editor(
        data,
        num_rows='dynamic',
        use_container_width=True,
        hide_index=True,
        column_config=education_config()
    )

    return data

def get_research_philanthropy(re_id):
    query = f'''
            SELECT
                "Foundation",
                "Date",
                "Amount",
                "Comments"
            FROM
                donor_research.philanthropy
            WHERE
                parent_id = '{re_id}' AND
                updated_on = (
                    SELECT
                        updated_on
                    FROM
                        donor_research.philanthropy
                    WHERE
                        parent_id = '{re_id}'
                    ORDER BY
                        updated_on DESC
                    LIMIT 1
                );
            '''

    try:
        data = fetch_data_from_sql(query)

    except SQLAlchemyError:
        data = pd.DataFrame(data={
            'Foundation': [None],
            'Date': [None],
            'Amount': [np.nan],
            'Comments': [None]
        })

    data = st.data_editor(
        data,
        num_rows='dynamic',
        use_container_width=True,
        hide_index=True,
        column_config=philanthropy_config()
    )

    return data

def get_research_location(re_id):
    query = f'''
            SELECT
                "Location"
            FROM
                donor_research.location
            WHERE
                parent_id = '{re_id}' AND
                updated_on = (
                    SELECT
                        updated_on
                    FROM
                        donor_research.location
                    WHERE
                        parent_id = '{re_id}'
                    ORDER BY
                        updated_on DESC
                    LIMIT 1
                );
            '''

    try:
        data = fetch_data_from_sql(query)

    except SQLAlchemyError:
        data = pd.DataFrame(data={
            'Location': [None]
        })

    data = st.data_editor(
        data,
        num_rows='dynamic',
        use_container_width=True,
        hide_index=True
    )

    return data

def get_research_net_worth(re_id):
    query = f'''
            SELECT
                "Net worth"
            FROM
                donor_research.net_worth
            WHERE
                parent_id = '{re_id}' AND
                updated_on = (
                    SELECT
                        updated_on
                    FROM
                        donor_research.net_worth
                    WHERE
                        parent_id = '{re_id}'
                    ORDER BY
                        updated_on DESC
                    LIMIT 1
                );
            '''

    try:
        data = fetch_data_from_sql(query)

    except SQLAlchemyError:
        data = pd.DataFrame(data={
            'Net worth': [None]
        })

    data = st.data_editor(
        data,
        num_rows='dynamic',
        use_container_width=True,
        hide_index=True
    )

    return data

def get_research_remarks(re_id):
    query = f'''
            SELECT
                "Remarks"
            FROM
                donor_research.remarks
            WHERE
                parent_id = '{re_id}' AND
                updated_on = (
                    SELECT
                        updated_on
                    FROM
                        donor_research.remarks
                    WHERE
                        parent_id = '{re_id}'
                    ORDER BY
                        updated_on DESC
                    LIMIT 1
                );
            '''

    try:
        data = fetch_data_from_sql(query)

    except SQLAlchemyError:
        data = pd.DataFrame(data={
            'Remarks': [None]
        })

    data = st.data_editor(
        data,
        num_rows='fixed',
        use_container_width=True,
        hide_index=True
    )

    return data

def employment_config():
    config = {
                'Organisation': st.column_config.TextColumn(
                    'Organisation'
                ),
                'Joining Year': st.column_config.NumberColumn(
                    'Joining Year',
                    format='%d',
                    min_value=1962,
                    max_value=int(date.today().strftime('%Y'))
                ),
                'End Year': st.column_config.NumberColumn(
                    'End Year',
                    format='%d',
                    min_value=1962,
                    max_value=int(date.today().strftime('%Y'))
                ),
                'Is Senior?': st.column_config.CheckboxColumn(
                    'Is Senior?',
                    default=False,
                ),
                'Stakes left': st.column_config.NumberColumn(
                    '% Stakes left',
                    min_value=0.00,
                    max_value=100,
                    step=0.01,
                    help='Percentage Stakes Left after the final stage / Hold currently'
                ),
                'Shares owned': st.column_config.NumberColumn(
                    'Shares owned',
                    min_value=0,
                    format='%d',
                    help='Number of shares owned if the company is Public'
                ),
                'Funding Stage': st.column_config.SelectboxColumn(
                    'Funding Stage',
                    default=None,
                    options=[
                        'Early Stage VC',
                        'Pre-seed',
                        'Seed',
                        'Series A',
                        'Series B',
                        'Series C',
                        'Series D',
                        'IPO',
                        'Exit',
                        'Acquired',
                        'Deadpooled'
                    ]
                ),
                'Company Valuation': st.column_config.NumberColumn(
                    'Valuation',
                    min_value=0.00,
                    format='$ %d M'
                ),
                'Share Price': st.column_config.NumberColumn(
                    'Share Price',
                    min_value=0.00,
                    format='₹ %d'
                ),
                'Salary': st.column_config.NumberColumn(
                    'Annual Salary',
                    min_value=0.00,
                    format='$ %d M'
                )
            }

    return config

def education_config():
    config={
        'University': st.column_config.TextColumn(
                'University'
        ),
        'Joining Year': st.column_config.NumberColumn(
            'Joined',
            format='%d',
            min_value=1962,
            max_value=int(date.today().strftime('%Y')) + 5
        ),
        'Graduated on': st.column_config.NumberColumn(
            'Left',
            format='%d',
            min_value=1962,
            max_value=int(date.today().strftime('%Y'))
        )
    }

    return config

def philanthropy_config():
    config = {
                'Date': st.column_config.DateColumn(
                    'Date',
                    format='MMM YYYY'
                ),
                'Amount': st.column_config.NumberColumn(
                    'Amount',
                    min_value=0.00,
                    format='₹ %d Cr.'
                )
            }

    return config

def capture_manual_data(re_id, update=False):
    st.subheader('')

    # Updating an existing record
    if update:
        st.subheader('Update/Modify research')
        employment, education, philanthropy, location, net_worth, remarks = get_research(re_id)

        # Submit button
        if st.button('Submit', type='primary', use_container_width=True):

            # Check if there's anything to upload
            if get_null_values(employment.drop(columns=['Is Senior?'])) == \
                    get_df_shape_product(employment.drop(columns=['Is Senior?'])) and \
                get_null_values(education) == get_df_shape_product(education) and \
                get_null_values(philanthropy) == get_df_shape_product(philanthropy) and \
                get_null_values(location) == get_df_shape_product(philanthropy) and \
                get_null_values(net_worth) == get_df_shape_product(net_worth) and \
                get_null_values(remarks) == get_df_shape_product(remarks):
                st.error('Nothing to update', icon='❗️')

            else:
                # Updating Employment
                if get_null_values(employment.drop(columns=['Is Senior?'])) != \
                        get_df_shape_product(employment.drop(columns=['Is Senior?'])):
                    employment['parent_id'] = re_id
                    employment['updated_on'] = datetime.now()
                    update_sql_table(employment, 'employment', schema='donor_research', replace=False)

                # Updating Location
                if get_null_values(location) != get_df_shape_product(location):
                    location['parent_id'] = re_id
                    location['updated_on'] = datetime.now()
                    update_sql_table(location, 'location', schema='donor_research', replace=False)

                # Updating Education
                if get_null_values(education) != get_df_shape_product(education):
                    education['parent_id'] = re_id
                    education['updated_on'] = datetime.now()
                    update_sql_table(education, 'education', schema='donor_research', replace=False)

                # Updating Philanthropy
                if get_null_values(philanthropy) != get_df_shape_product(philanthropy):
                    philanthropy['parent_id'] = re_id
                    philanthropy['updated_on'] = datetime.now()
                    update_sql_table(philanthropy, 'philanthropy', schema='donor_research', replace=False)

                # Updating Remarks
                if get_null_values(remarks) != get_df_shape_product(remarks):
                    remarks['parent_id'] = re_id
                    remarks['updated_on'] = datetime.now()
                    update_sql_table(remarks, 'remarks', schema='donor_research', replace=False)

                # Updating Networth
                if get_null_values(net_worth) != get_df_shape_product(net_worth):
                    net_worth['parent_id'] = re_id
                    net_worth['updated_on'] = datetime.now()
                    update_sql_table(net_worth, 'remarks', schema='donor_research', replace=False)

                st.cache_data.clear()

                st.toast('Data Updated Successfully!', icon='✅')
                time.sleep(1)
                st.rerun()

    else:
        st.subheader('Manual research')

        manual_employment = st.data_editor(
            save_manual_employment(re_id=re_id),
            num_rows='dynamic',
            use_container_width=True,
            hide_index=True,
            column_config=employment_config()
        )

        col1, col2 = st.columns(2)

        with col1:
            st.write('**Education**')
            manual_education = st.data_editor(
                save_manual_education(re_id),
                num_rows='dynamic',
                use_container_width=True,
                hide_index=True,
                column_config=education_config()
            )

        with col2:

            st.write('**Philanthropy**')
            manual_philanthropy = st.data_editor(
                save_manual_philanthropy(),
                num_rows='dynamic',
                use_container_width=True,
                hide_index=True,
                column_config=philanthropy_config()
            )

            col3, col4 = st.columns(2)

            with col3:
                st.write('**Location**')
                manual_location = st.data_editor(
                    save_manual_location(),
                    use_container_width=True,
                    hide_index=True
                )

            with col4:
                st.write('**Net Worth**')
                net_worth = st.data_editor(
                    save_manual_net_worth(),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Net worth': st.column_config.NumberColumn(
                            'Net worth',
                            min_value=0.00,
                            format='$ %d Million'
                        )
                    }
                )

        st.write('**Remarks**')
        remarks = st.text_area('Remarks', label_visibility='collapsed')

        # Submit button
        if st.button('Submit', type='primary', use_container_width=True):

            remarks = pd.DataFrame(data={'Remarks': [remarks]})

            if get_null_values(manual_employment.drop(columns=['Is Senior?'])) == \
                    get_df_shape_product(manual_employment.drop(columns=['Is Senior?'])) \
                    and get_null_values(manual_education) == get_df_shape_product(manual_education) \
                    and get_null_values(manual_location) == get_df_shape_product(manual_location) \
                    and get_null_values(manual_philanthropy) == get_df_shape_product(manual_philanthropy) \
                    and get_null_values(remarks) == get_df_shape_product(remarks) \
                    and get_null_values(net_worth) == get_df_shape_product(net_worth):
                st.error('No data to upload', icon='❗️')

            else:

                # Update the SQL table with employment data
                st.toast('Uploading Data', icon='⬆️')
                time.sleep(1)

                # Updating Employment
                if get_null_values(manual_employment.drop(columns=['Is Senior?'])) != \
                        get_df_shape_product(manual_employment.drop(columns=['Is Senior?'])):
                    manual_employment['parent_id'] = re_id
                    manual_employment['updated_on'] = datetime.now()
                    update_sql_table(manual_employment, 'employment', schema='donor_research', replace=False)

                # Updating Location
                if get_null_values(manual_location) != get_df_shape_product(manual_location):
                    manual_location['parent_id'] = re_id
                    manual_location['updated_on'] = datetime.now()
                    update_sql_table(manual_location, 'location', schema='donor_research', replace=False)

                # Updating Education
                if get_null_values(manual_education) != get_df_shape_product(manual_education):
                    manual_education['parent_id'] = re_id
                    manual_education['updated_on'] = datetime.now()
                    update_sql_table(manual_education, 'education', schema='donor_research', replace=False)

                # Updating Philanthropy
                if get_null_values(manual_philanthropy) != get_df_shape_product(manual_philanthropy):
                    manual_philanthropy['parent_id'] = re_id
                    manual_philanthropy['updated_on'] = datetime.now()
                    update_sql_table(manual_philanthropy, 'philanthropy', schema='donor_research', replace=False)

                # Updating Remarks
                if get_null_values(remarks) != get_df_shape_product(remarks):
                    remarks['parent_id'] = re_id
                    remarks['updated_on'] = datetime.now()
                    update_sql_table(remarks, 'remarks', schema='donor_research', replace=False)

                # Updating Networth
                if get_null_values(net_worth) != get_df_shape_product(net_worth):
                    net_worth['parent_id'] = re_id
                    net_worth['updated_on'] = datetime.now()
                    update_sql_table(net_worth, 'remarks', schema='donor_research', replace=False)

                st.toast('Data Uploaded Successfully!', icon='✅')
                time.sleep(1)

                # Clearing Cache
                st.toast('Clearing Cache', icon='🗑️')
                time.sleep(1)
                st.cache_data.clear()

                # Reloading Page
                st.toast('Reloading Page', icon='🔄')
                time.sleep(1)

                st.toast('Please proceed now', icon='🙏🏻')
                time.sleep(1)
                st.rerun()

# Streamlit form to edit and submit data
def main():

    with st.sidebar:

        # Add Title
        st.title('Donor Research')
        st.divider()

        # Get a drop-down list of prospects
        st.header('Select Prospect')

        update_existing = st.toggle('View researched prospects?')

        # Get Prospects
        prospects = get_prospects(update_existing)

        prospect = st.selectbox(
            'Prospects',
            prospects,
            placeholder="Select a Prospect...",
            index=None,
            label_visibility = 'hidden'
        )

        st.divider()

        # Upload data from Live Alumni
        st.header('Upload Data from Live Alumni')
        uploaded_file = st.file_uploader(
            'Upload Files',
            type=['csv'],
            label_visibility='hidden'
        )

        if uploaded_file:
            with st.status('Uploading Live Alumni Data...', expanded=True) as status:
                process_live_alumni(uploaded_file)
                status.update(
                    label='Live Alumni Data uploaded successfully!',
                    state="complete",
                    expanded=False
                )

            st.stop()

        # Clear Cache
        if st.button("Clear Cache"):
            # Clear values from *all* all in-memory and on-disk data caches:
            # i.e. clear values from both square and cube
            st.cache_data.clear()

    # On selection
    if prospect:
        # Display RE Data
        system_record_id = display_re_data(prospect)

        # Display Live Alumni Data
        display_live_alumni_data(system_record_id)

        # Display editable data in Streamlit
        capture_manual_data(system_record_id, update_existing)

if __name__ == '__main__':
    st.set_page_config(
        page_title='Donor Research',
        page_icon=':bulb:',
        layout='wide'
    )

    hide_style = '''
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    '''

    st.markdown(hide_style, unsafe_allow_html=True)

    main()
