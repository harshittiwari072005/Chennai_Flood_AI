from fastapi import APIRouter

router = APIRouter()

ZONES = [
    {"id":"Z01","name":"Velachery",       "lat":12.9788,"lon":80.2209,"risk":83.3,"class":"critical"},
    {"id":"Z02","name":"Adyar Riverbank", "lat":13.0012,"lon":80.2565,"risk":77.0,"class":"critical"},
    {"id":"Z03","name":"Tambaram",        "lat":12.9249,"lon":80.1000,"risk":55.2,"class":"high"},
    {"id":"Z04","name":"Mudichur",        "lat":12.8938,"lon":80.0641,"risk":66.6,"class":"high"},
    {"id":"Z05","name":"Saidapet",        "lat":13.0211,"lon":80.2244,"risk":64.8,"class":"high"},
    {"id":"Z06","name":"Porur",           "lat":13.0358,"lon":80.1568,"risk":49.2,"class":"medium"},
    {"id":"Z07","name":"Kolathur",        "lat":13.1165,"lon":80.2090,"risk":56.2,"class":"high"},
    {"id":"Z08","name":"Perambur",        "lat":13.1128,"lon":80.2455,"risk":59.1,"class":"high"},
    {"id":"Z09","name":"Anna Nagar",      "lat":13.0878,"lon":80.2101,"risk":43.0,"class":"medium"},
    {"id":"Z10","name":"T Nagar",         "lat":13.0418,"lon":80.2341,"risk":51.3,"class":"high"},
    {"id":"Z11","name":"Ambattur",        "lat":13.1142,"lon":80.1544,"risk":43.0,"class":"medium"},
    {"id":"Z12","name":"Sholinganallur",  "lat":12.9010,"lon":80.2279,"risk":63.9,"class":"high"},
    {"id":"Z13","name":"Nungambakkam",    "lat":13.0580,"lon":80.2474,"risk":22.0,"class":"low"},
    {"id":"Z14","name":"Alwarpet",        "lat":13.0320,"lon":80.2568,"risk":34.7,"class":"medium"},
    {"id":"Z15","name":"Marina North",    "lat":13.0827,"lon":80.2800,"risk":68.3,"class":"high"},
]

@router.get("/zones")
def get_zones():
    return {"zones": ZONES, "total": len(ZONES)}