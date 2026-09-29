"""
Corpus SYNTHÉTIQUE de SMS signalés par les clients à un numéro court « arnaque ».

Aucun jeu public de SMS d'arnaque Mobile Money congolais n'existe. Ce générateur reproduit
les arnaques documentées par les opérateurs et les régulateurs de la région (faux « envoi
par erreur », faux agent ou faux service client, faux gain, demande de code), et des
messages LÉGITIMES que des clients transfèrent aussi par doute : vraies confirmations de
l'opérateur (qui ressemblent aux fausses), messages familiaux, promotions, tontines.

Français majoritaire, avec des messages en lingala et en swahili. ATTENTION : les phrases
en langues nationales sont des approximations à faire valider par des locuteurs natifs
avant tout usage réel.

Chaque modèle de message porte un identifiant : l'évaluation sépare les modèles entre
apprentissage et test (un classifieur ne doit pas seulement reconnaître des phrases vues).
"""
from __future__ import annotations

import random
import re

CATEGORIES = ["ENVOI_ERREUR", "FAUX_AGENT", "FAUX_GAIN", "DEMANDE_CODE", "LEGITIME"]
SCAM = [c for c in CATEGORIES if c != "LEGITIME"]
LABELS = {
    "ENVOI_ERREUR": "Faux « envoi par erreur »", "FAUX_AGENT": "Faux agent / faux service client",
    "FAUX_GAIN": "Faux gain / loterie", "DEMANDE_CODE": "Demande de code ou de PIN", "LEGITIME": "Message légitime",
}

OPS = [("Vodacom", "M-Pesa"), ("Airtel", "Airtel Money"), ("Orange", "Orange Money")]

# (identifiant, langue, expéditeur, gabarit). Expéditeur : "numero" (un particulier) ou "operateur".
TEMPLATES: dict[str, list[tuple[str, str, str, str]]] = {
    "ENVOI_ERREUR": [
        ("ee1", "fr", "numero", "Bonjour {titre}, je viens de vous envoyer {montant} par erreur. C'était pour payer l'hôpital de ma mère. {svp} renvoyez-moi au {num}. Que Dieu vous bénisse."),
        ("ee2", "fr", "numero", "Excusez-moi, je me suis trompé de numéro en envoyant {montant}. Merci de me retourner l'argent au {num}, c'est urgent."),
        ("ee3", "fr", "numero", "Bonsoir, transfert de {montant} effectué sur votre numéro par erreur. Veuillez le rembourser au {num} avant que je porte plainte à la police."),
        ("ee4", "fr", "numero", "Pardon {titre}, erreur de transfert : {montant} envoyés chez vous au lieu de mon fournisseur. Renvoyez au {num}, je vous laisse 1000 FC pour le dérangement."),
        ("ee5", "fr", "numero", "{titre2}, j'ai envoyé {montant} sur ton compte par erreur, renvoie {stp} au {num}, je suis bloqué à l'hôpital."),
        ("ee6", "fr", "numero", "Vous avez reçu {montant} de ma part par erreur, c'était destiné à mon frère. Renvoyez vite au {num} svp, que Dieu vous garde."),
        ("ee7", "ln", "numero", "Mbote! Natindeli yo mbongo {montant} na libunga. {svp} zongisa yango na {num}. Merci mingi."),
        ("ee8", "sw", "numero", "Habari, nimekutumia pesa {montant} kwa makosa. Tafadhali nirudishie kwa namba {num}. Asante sana."),
        ("ee9", "fr", "numero", "Allô, c'est moi qui viens de t'envoyer {montant} par erreur, je voulais payer le loyer. Renvoie {stp} au {num}."),
        ("ee10", "fr", "numero", "Bjr, mon enfant a envoyé {montant} sur votre numéro en jouant avec mon téléphone. Remboursez {svp} au {num}."),
        ("ee11", "fr", "numero", "ATTENTION : {montant} vous ont été crédités par erreur. Remboursez au {num} sinon votre compte {money} sera bloqué."),
        ("ee12", "fr", "numero", "Je suis {nom}, j'ai fait une fausse manipulation et l'argent ({montant}) est parti chez vous. Aidez-moi, renvoyez au {num}."),
        ("ee13", "ln", "numero", "Ndeko, mbongo {montant} ekoti na compte na yo na libunga. Zongisela ngai na {num} {svp}."),
        ("ee14", "sw", "numero", "Ndugu, pesa {montant} imeingia kwako kwa bahati mbaya. Nirudishie kwa {num} tafadhali."),
    ],
    "FAUX_AGENT": [
        ("fa1", "fr", "numero", "{op} : votre compte {money} sera suspendu dans 24h pour vérification KYC. Appelez notre agent au {num} pour le réactiver."),
        ("fa2", "fr", "numero", "Cher client {op}, une anomalie a été détectée sur votre portefeuille. Appelez le service technique au {num} et gardez votre PIN à portée de main."),
        ("fa3", "fr", "numero", "Service {money} : votre compte est bloqué. Pour le débloquer, envoyez {frais} de frais au {num}."),
        ("fa4", "fr", "numero", "Agent {op} ici : mise à jour obligatoire de votre carte SIM. Envoyez le code reçu par SMS au {num}."),
        ("fa5", "fr", "numero", "{op} INFO : suite à la nouvelle loi, votre compte {money} doit être revalidé aujourd'hui. Contactez le conseiller au {num}."),
        ("fa6", "ln", "numero", "Mbote, awa ezali service client {op}. Compte na yo ekangami, benga agent na {num}."),
        ("fa7", "sw", "numero", "Huduma kwa wateja {op}: akaunti yako ya {money} imefungwa. Piga simu kwa wakala {num} ili kuifungua."),
        ("fa8", "fr", "numero", "Votre ligne {op} sera désactivée ce soir faute d'identification. Pour l'éviter, appelez le centre au {num}."),
        ("fa9", "fr", "numero", "Bonjour, ici le superviseur des agents {money}. Votre compte présente un double débit, rappelez-nous au {num} pour le remboursement."),
        ("fa10", "fr", "numero", "{money} : pour bénéficier du nouveau plafond, confirmez votre identité auprès de l'agent {agent} au {num}."),
        ("fa11", "sw", "numero", "Mpendwa mteja, laini yako ya {op} itafungwa leo. Piga {num} kuthibitisha kitambulisho chako."),
    ],
    "FAUX_GAIN": [
        ("fg1", "fr", "numero", "Félicitations ! Votre numéro a gagné {gain} au grand tirage {op}. Envoyez les frais de dossier de {frais} au {num} pour recevoir votre gain."),
        ("fg2", "fr", "numero", "BRAVO ! Vous êtes l'heureux gagnant d'une moto et de {gain}. Contactez le responsable au {num}."),
        ("fg3", "fr", "numero", "Promo {op} : vous avez gagné {gain}. Pour valider, appelez le {num} et payez {frais} de frais d'envoi."),
        ("fg4", "fr", "numero", "Tombola : votre carte SIM a été tirée au sort, gain de {gain}. Appelez vite le {num}, offre valable aujourd'hui seulement."),
        ("fg5", "fr", "numero", "Votre numéro a été sélectionné pour une bourse de {gain} de la fondation. Réservez votre place en envoyant {frais} au {num}."),
        ("fg6", "sw", "numero", "Hongera! Umeshinda {gain} kwenye bahati nasibu ya {op}. Tuma ada ya {frais} kwa {num}."),
        ("fg7", "fr", "numero", "Le programme d'aide du gouvernement vous accorde {gain}. Payez {frais} de frais de transfert au {num} pour le recevoir."),
        ("fg8", "fr", "numero", "Vous avez gagné un téléphone neuf et {gain} de crédit. Contactez notre service des lots au {num}."),
        ("fg9", "fr", "numero", "Jeu {op} : 3 gagnants ce mois, vous en faites partie ({gain}). Envoyez {frais} au {num} pour la livraison."),
        ("fg10", "ln", "numero", "Felicitations! Olongi {gain} na tombola ya {op}. Tinda {frais} na {num} mpo na kozwa yango."),
    ],
    "DEMANDE_CODE": [
        ("dc1", "fr", "numero", "Bonjour, je vous ai envoyé un code par erreur, pouvez-vous me le transférer au {num} {svp} ? C'est urgent."),
        ("dc2", "fr", "numero", "{op} sécurité : pour annuler une transaction suspecte sur votre compte, envoyez votre code PIN au {num}."),
        ("dc3", "fr", "numero", "Votre code de vérification {money} est nécessaire pour recevoir {montant}. Communiquez-le à l'agent au {num}."),
        ("dc4", "fr", "numero", "Un code à 6 chiffres vient d'arriver sur ton téléphone, c'est le mien, envoie-le moi {stp} au {num}."),
        ("dc5", "ln", "numero", "Mbote, code moko eyei na telefone na yo na libunga. Tindela ngai yango na {num} {svp}."),
        ("dc6", "sw", "numero", "Samahani, nimetuma msimbo kwa simu yako kimakosa. Tafadhali nitumie kwa {num}."),
        ("dc7", "fr", "numero", "Maman c'est moi, j'ai changé de numéro. Un code va arriver sur ton téléphone, envoie-le vite au {num}."),
        ("dc8", "fr", "numero", "Pour recevoir votre remboursement {money}, communiquez le code de validation reçu au {num}."),
        ("dc9", "fr", "numero", "Service client {op} : votre PIN a expiré. Répondez à ce message avec votre ancien PIN ou appelez le {num}."),
        ("dc10", "sw", "numero", "Wewe ni mshindi! Ili kupokea zawadi, nitumie msimbo uliopokea kwa {num}."),
    ],
    "LEGITIME": [
        ("lg1", "fr", "operateur", "Vous avez reçu {montant} de {num} ({nom}). Nouveau solde : {solde}. Réf : {ref}."),
        ("lg2", "fr", "operateur", "Transfert de {montant} vers {num} réussi. Frais : {frais}. Nouveau solde : {solde}. {money}"),
        ("lg3", "fr", "operateur", "{op} : profitez de 1 Go à 500 FC valable 24h. Composez *{code}#."),
        ("lg4", "fr", "operateur", "Votre retrait de {montant} chez l'agent {agent} a été effectué. Solde : {solde}. Merci d'utiliser {money}."),
        ("lg5", "fr", "operateur", "Votre facture d'électricité de {montant} a été payée. Référence : {ref}. {money}"),
        ("lg6", "fr", "numero", "Maman, je suis bien arrivé à Kinshasa. Rappelle-moi ce soir au {num}."),
        ("lg7", "fr", "numero", "Rappel : réunion de la tontine dimanche à 15h chez Mama {nom}. Cotisation {montant}."),
        ("lg8", "fr", "numero", "Bonjour, c'est {nom}, j'ai bien reçu ton argent, merci beaucoup ! On se voit samedi."),
        ("lg9", "fr", "numero", "Rendez-vous au centre de santé confirmé pour demain 9h. Pour annuler, appelez le {num}."),
        ("lg10", "fr", "operateur", "{op} : votre forfait appels expire demain. Renouvelez en composant *{code}#."),
        ("lg11", "sw", "numero", "Habari, nimefika salama. Nitakupigia kesho asubuhi."),
        ("lg12", "ln", "numero", "Mbote mama, tokutani lobi na ndako. Nazali malamu."),
        ("lg13", "fr", "numero", "Le camion part demain à 6h pour Lubumbashi, sois à l'heure. Mon numéro : {num}."),
        ("lg14", "fr", "operateur", "Votre paiement de {montant} chez {marchand} a été accepté. Nouveau solde : {solde}."),
        ("lg15", "fr", "operateur", "Vous avez envoyé {montant} à {nom} ({num}). Frais : {frais}. Solde disponible : {solde}."),
        ("lg16", "fr", "operateur", "{money} : votre code de confirmation est {ref}. Ne le communiquez à personne, pas même à un agent."),
        ("lg17", "fr", "operateur", "Dépôt de {montant} reçu chez l'agent {agent}. Nouveau solde : {solde}. {money}"),
        ("lg18", "fr", "numero", "Papa, j'ai payé les frais scolaires ce matin, le reçu est avec moi. Bonne journée."),
        ("lg19", "fr", "numero", "Salut, c'est {nom}. Je t'ai envoyé {montant} pour la cotisation du mariage, confirme quand tu reçois."),
        ("lg20", "fr", "numero", "Le marché est fermé demain, on se retrouve lundi. Mon nouveau numéro de travail est le {num}."),
        ("lg21", "sw", "operateur", "Umepokea {montant} kutoka {num}. Salio jipya ni {solde}. {money}"),
        ("lg22", "sw", "numero", "Mama, nimekutumia pesa ya shule {montant}. Tafadhali thibitisha ukipokea."),
        ("lg23", "sw", "numero", "Mkutano wa kikundi utakuwa Jumapili saa tisa. Mchango ni {montant}."),
        ("lg24", "ln", "numero", "Tata, natindeli yo mbongo ya kelasi {montant}. Yebisa ngai soki ekoti."),
        ("lg25", "ln", "numero", "Mbote, nakoya kotala bino lobi na mpokwa. Bokeba nzela."),
        ("lg26", "ln", "operateur", "Ozwi mbongo {montant} ya {num}. Solde na yo sika : {solde}. {money}"),
    ],
}

NAMES = ["Kabeya", "Mbuyi", "Nsimba", "Mukendi", "Ilunga", "Kasongo", "Lukusa", "Tshibanda", "Mwamba", "Ngalula"]
MERCHANTS = ["Pharmacie du Centre", "Supermarché Kin Marché", "Station Total Gombe", "Boutique Mama Nzuzi"]


def _number(rng: random.Random, obfuscate: bool) -> str:
    prefix = rng.choice(["81", "82", "83", "84", "85", "89", "97", "98", "99", "90", "91"])
    rest = f"{rng.randrange(10**7):07d}"
    digits = prefix + rest
    style = rng.random()
    if obfuscate and style < 0.35:
        return " ".join("0" + digits)                                 # 0 8 1 2 3 …
    if style < 0.25:
        return f"+243 {digits[:2]} {digits[2:5]} {digits[5:]}"
    if style < 0.45:
        return f"0{digits[:2]}-{digits[2:5]}-{digits[5:7]}-{digits[7:]}"
    if style < 0.6:
        return f"(+243) {digits}"
    if style < 0.7:
        return f"243{digits}"
    return f"0{digits}"


def _amount(rng: random.Random) -> str:
    if rng.random() < 0.6:
        v = rng.choice([5, 10, 15, 20, 25, 50, 75, 100, 150]) * 1000
        return rng.choice([f"{v:,} FC".replace(",", " "), f"{v} FC", f"{v:,} CDF".replace(",", " "), f"{v}fc"])
    v = rng.choice([10, 20, 30, 50, 100, 200])
    return rng.choice([f"{v} $", f"{v} USD", f"{v}$"])


ABBREV = [(r"\bs'il vous plaît\b", "svp"), (r"\bs'il te plaît\b", "stp"), (r"\bpour\b", "pr"), (r"\bmerci\b", "mrc"),
          (r"\baujourd'hui\b", "ajd"), (r"\bbeaucoup\b", "bcp"), (r"\bvous\b", "vs"), (r"\bquelque\b", "qlq")]


def _noise(text: str, rng: random.Random) -> str:
    """Écriture SMS : abréviations, accents perdus, fautes de frappe, casse."""
    if rng.random() < 0.4:
        for pat, rep in ABBREV:
            if rng.random() < 0.5:
                text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    if rng.random() < 0.3:
        text = text.translate(str.maketrans("éèêàâçôîù", "eeeaacoiu"))
    if rng.random() < 0.25:
        chars = list(text)
        for _ in range(rng.randint(1, 3)):
            i = rng.randrange(1, max(2, len(chars) - 1))
            if chars[i].isalpha() and chars[i - 1].isalpha():
                chars[i - 1], chars[i] = chars[i], chars[i - 1]
        text = "".join(chars)
    if rng.random() < 0.12:
        text = text.upper()
    elif rng.random() < 0.15:
        text = text.lower()
    return text


def render(template: str, rng: random.Random, number: str | None = None, obfuscate: bool = False) -> tuple[str, str]:
    """Texte du message et numéro (forme d'affichage) qu'il contient."""
    op, money = rng.choice(OPS)
    num = number or _number(rng, obfuscate)
    fields = {
        "titre": rng.choice(["Monsieur", "Madame", "Papa", "Maman", "cher client", "frère"]),
        "titre2": rng.choice(["Grand frère", "Maman", "Tantine", "Papa", "Ya"]),
        "svp": rng.choice(["Svp", "S'il vous plaît", "stp", "Svp svp"]), "stp": rng.choice(["stp", "s'il te plaît", "vite"]),
        "montant": _amount(rng), "frais": _amount(rng), "gain": rng.choice(["500 000 FC", "1 000 $", "2 000 000 FC", "250 $", "5 000 $"]),
        "solde": _amount(rng), "num": num, "op": op, "money": money, "nom": rng.choice(NAMES),
        "code": rng.choice(["123", "144", "1222", "555", "100"]), "ref": f"{rng.choice('ABCDEFGH')}{rng.randrange(10**8):08d}",
        "agent": f"A{rng.randrange(10**6):06d}", "marchand": rng.choice(MERCHANTS),
    }
    return _noise(template.format(**fields), rng), num


def generate(n: int = 6000, seed: int = 7, scam_share: float = 0.62) -> list[dict]:
    """n messages signalés : environ 62 % d'arnaques (hypothèse : la plupart des signalements sont
    fondés, mais une part notable sont des doutes sur des messages légitimes)."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        cat = rng.choice(SCAM) if rng.random() < scam_share else "LEGITIME"
        tid, lang, sender_kind, tpl = rng.choice(TEMPLATES[cat])
        text, num = render(tpl, rng, obfuscate=cat != "LEGITIME" and rng.random() < 0.15)
        if sender_kind == "operateur":
            sender = rng.choice(["VODACOM", "M-PESA", "AIRTEL", "AirtelMoney", "ORANGE", "OrangeMoney"])
        else:   # un escroc écrit souvent depuis un autre numéro que celui où il demande l'argent
            sender = num if rng.random() < 0.5 else _number(rng, False)
        rows.append({"id": f"S{i:05d}", "category": cat, "template": tid, "lang": lang,
                     "sender": sender, "text": text})
    return rows
