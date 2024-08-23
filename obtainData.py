import requests;
import dotenv;
import os;


    

class ObtainData():
    def __init__(self,email:str,password:str):
        self.email =email
        self.password = password
        self.session = requests.Session()
        return
    
         
    def login(self) -> Exception|None:
        url = "https://mein-senec.de/auth/login"
        headers = {"Content-Type":"application/x-www-form-urlencoded",
                   "Referer":"https://mein-senec.de/auth/login",
                   "Origin":"https://mein-senec.de",
                   "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
        resp = self.session.post(url,data={"username":self.email,"password":self.password},headers=headers)
        if resp.status_code != 200:
            return Exception(f"Failed to login: {resp.text}")
        return None


    def get_all_file_names(self)->tuple[dict[str,str]| None,Exception|None]:
        url = "https://mein-senec.de/endkunde/api/statistischeDaten/getKalenderWochen?anlageNummer=0"
        headers= {"Referer":"https://mein-senec.de/endkunde/",
                  "Origin":"https://mein-senec.de",
                  "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
        resp = self.session.get(url,headers=headers)
        if resp.status_code != 200:
            return (None,Exception(f"Failed to get file names status code: {resp.status_code}"))
        return (resp.json(),None)
    def get_files_contents(self,kw:str,year:str)->tuple[str,Exception|None]:
        headers= {"Referer":"https://mein-senec.de/endkunde/",
                  "Origin":"https://mein-senec.de",
                  "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
        url = f"https://mein-senec.de/endkunde/api/statistischeDaten/download?anlageNummer=0&woche={kw}&jahr={year}"
        resp = self.session.get(url,headers=headers)
        if resp.status_code!= 200:
            return (None,Exception(f"Failed to get file contents status code: {resp.status_code}"))
        return (resp.text,None)
    def flow(self)->tuple[None,Exception|None]:
        # login
        resp = self.login()
        if resp != None:
            return (None,resp)

        # get the export file names / parameters to construct those filenames
        file_names,err = self.get_all_file_names()
        if err != None:
            return (None,err)
        if file_names == None:
            return (None,Exception("Failed to get file names"))
        for params in file_names:
            kw = params["kalenderwoche"]
            year = params["jahr"]
            (contents,err) = self.get_files_contents(kw,year)
            if err != None:
                # will add loggin in the future
                continue
            file = open(f"data/solar_data_kw_{kw}_jahr_{year}.csv","w")
            file.write(contents)
            file.flush()
            file.close()
        return (None,None)




dotenv.load_dotenv()
data_getter =ObtainData(os.environ["Senec_Email"],os.environ["Senec_Pass"])
data_getter.flow()
