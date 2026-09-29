"""scheduled fomc decision dates (last day of each meeting) -> data/fomc.csv"""
import re, requests, pandas as pd
h = {"User-Agent": "Mozilla/5.0"}
MON = "Jan(?:uary)?|Feb(?:ruary)?|March|April|May|June|July|August|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"


def txt(u):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", requests.get(u, headers=h, timeout=30).text).replace("\xa0", " "))


hist = re.compile(rf"((?:{MON})(?:/(?:{MON}))?) (\d{{1,2}})(?:-(\d{{1,2}}))? Meeting - (\d{{4}})")
new = re.compile(rf"((?:{MON})(?:/(?:{MON}))?) (\d{{1,2}})(?:-(\d{{1,2}}))?\*? Statement:")
out = []
for y in range(1994, 2021):
    for m, d1, d2, yy in hist.findall(txt(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm")):
        if int(yy) == y:
            out.append(pd.Timestamp(f"{m.split('/')[-1][:3]} {d2 or d1} {y}"))
t = txt("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
for y in range(2021, 2027):
    i = t.find(f"{y} FOMC Meetings")
    j = t.find(f"{y - 1} FOMC Meetings")
    seg = t[i:j] if j > i else t[i:]
    for m, d1, d2 in new.findall(seg):
        out.append(pd.Timestamp(f"{m.split('/')[-1][:3]} {d2 or d1} {y}"))
s = pd.Series(sorted(set(out)))
s.to_csv("data/fomc.csv", index=False, header=["date"])
print(len(s), s.groupby(s.dt.year).size().to_dict())
