# 📜 SSAFY AI 챌린지 규칙

> 본 규칙을 준수하여 대회에 참여해 주시기 바랍니다. 규칙 위반 시 불이익이 발생할 수 있으니 유의하시기 바랍니다.

## ⚡ 핵심 요약

| 구분 | 요점 |
|------|------|
| 👥 팀 | 4~5명, 대회 기간 중 **팀 변경 불가** |
| 📦 데이터 | 외부 데이터 **사용 가능** (공개 데이터만), 증강 허용, **Data Leakage 금지** |
| 🤖 모델 | Hugging Face 사전 학습 **VQA 모델**, LoRA/양자화 허용 |
| 🚫 금지 | **API 호출 방식 추론 엄격히 금지**, 테스트 데이터 사전 접근·유출 |
| 📤 제출 | 팀당 **하루 최대 20회**, CSV + 모델 코드 + 추가 문서 |
| 🔁 재현성 | 코드 실행 시 **동일한 결과**가 나와야 함 |

---

# Part 1. 대회 규칙 (SSAFY)

## 1. 참가 규정

- **참가 대상**: SSAFY 16기 교육생으로 구성된 팀
- **팀 구성**: 각 팀은 **4~5명**으로 구성되며, 팀원 간 협업을 통해 문제를 해결해야 합니다.
- **참가 제한**: 대회 기간 동안 **팀 변경은 불가**합니다.

## 2. 데이터 사용 규정

- **제공 데이터셋**
  - 학습용 데이터는 모델 학습에 사용 가능합니다.
  - 데이터셋은 대회 주최 측에서 제공하며, **외부 데이터 사용도 가능**합니다.
- **데이터 증강**
  - 데이터 증강은 허용됩니다.
  - 단, **공개적으로 사용 가능한 데이터**여야 합니다.
- **Data Leakage 금지**
  - 테스트 데이터에 대한 사전 접근 또는 유출은 **엄격히 금지**됩니다.

## 3. 모델 개발 규정

- **사용 가능 모델**
  - 허깅페이스(Hugging Face)에서 제공하는 사전 학습 모델 사용 가능
  - 대회 주제와 관련된 **Visual Question Answering(VQA) 모델**을 활용해야 합니다.
- **Fine-tuning**
  - 제공된 데이터셋을 활용하여 모델을 학습시키는 과정은 허용됩니다.
  - LoRA, 양자화 등 효율적인 학습 기법을 사용할 수 있습니다.
- **프롬프트 엔지니어링**
  - 모델의 성능을 향상시키기 위한 다양한 프롬프트 설계 기법을 적용할 수 있습니다.
- **API 통신**
  - ⛔ API 호출 및 응답을 이용하여 추론하는 방식은 **엄격히 금지**됩니다.

## 4. 제출 규정

- **제출 파일**

  | 파일 | 내용 |
  |------|------|
  | CSV 파일 | 각 이미지에 대한 질문과 답변 포함 |
  | 모델 코드 | 학습 및 추론에 사용된 코드 파일 |
  | 추가 문서 | 데이터 처리 및 모델 개발 과정에 대한 설명 포함 |

- **제출 횟수**: 팀당 하루 최대 **20회**로 제한됩니다.
- **재현성**: 제출된 솔루션은 재현 가능해야 하며, 코드 실행 시 동일한 결과가 나와야 합니다.

## 5. 평가 규정

| 항목 | 내용 |
|------|------|
| **리더보드 점수** | 모델의 정확도(Accuracy)를 기준으로 점수 산정<br>리더보드는 실시간으로 업데이트되며, 순위에 따라 점수 제공 |
| **결과 검증** | 제출된 솔루션은 대회 종료 후 주최 측에서 검증 |
| **Discussion 점수** | 글이나 댓글 작성을 토대로 별도의 점수 추가 부여 |
| **발표 점수** | 별도의 심사 기준에 따라 발표 점수 제공 |

## 6. 기타 규정

- **공정성**: 모든 참가자는 공정하게 대회에 참여해야 하며, 부정행위가 적발될 경우 **즉시 실격** 처리됩니다.
- **소통 활성화**: 디스커션 게시판을 적극 활용하여 팀 간 소통 및 아이디어 공유를 장려합니다.
- **문의**: 대회 관련 문의는 주최 측에서 제공하는 공식 채널을 통해 가능합니다.

---

# Part 2. Kaggle Competition Foundational Rules

> **Non-editable.** 대회 참가자는 Kaggle의 Foundational Competition Rules에도 동의해야 하며, 두 규칙이 충돌할 경우 **Kaggle 규칙이 우선**합니다.

Competition participants must also agree to Kaggle's Foundational Competition Rules. These rules will supersede the competition-specific rules in the event of any conflict.

The following Kaggle Competition Foundational Rules (**"Foundational Rules"**) apply to every competition regardless of whether the Sponsor creates competition-specific rules. Any competition-specific rules provided by the Sponsor are in addition to these rules, and in the case of any conflict or inconsistency, these Foundational Rules control and nullify contrary competition-specific rules.

## 📑 목차

| # | 조항 | # | 조항 |
|---|------|---|------|
| 1 | [Eligibility](#1-eligibility) | 10 | [Taxes](#10-taxes) |
| 2 | [Sponsor and Hosting Platform](#2-sponsor-and-hosting-platform) | 11 | [General Conditions](#11-general-conditions) |
| 3 | [Competition Period](#3-competition-period) | 12 | [Publicity](#12-publicity) |
| 4 | [Competition Entry](#4-competition-entry) | 13 | [Privacy](#13-privacy) |
| 5 | [Individuals and Teams](#5-individuals-and-teams) | 14 | [Warranty, Indemnity and Release](#14-warranty-indemnity-and-release) |
| 6 | [Submission Code Requirements](#6-submission-code-requirements) | 15 | [Internet](#15-internet) |
| 7 | [Determining Winners](#7-determining-winners) | 16 | [Right to Cancel, Modify or Disqualify](#16-right-to-cancel-modify-or-disqualify) |
| 8 | [Notification of Winners & Disqualification](#8-notification-of-winners--disqualification) | 17 | [Not an Offer or Contract of Employment](#17-not-an-offer-or-contract-of-employment) |
| 9 | [Prizes](#9-prizes) | 18 | [Definitions](#18-definitions) |

### 🔑 핵심 포인트 (한글 요약)

- **계정**: 1인 1계정. 다중 계정으로 제출하면 실격 (§5)
- **팀**: 하나의 팀에만 소속 가능. 팀 밖으로 코드·데이터 **비공개 공유 금지** (§5, §6)
- **코드 공유**: 공개 공유는 가능하나, 대회 포럼/노트북에 올려야 함 (§6)
- **오픈소스**: OSI 승인 라이선스 + 상업적 이용을 제한하지 않는 것만 사용 (§6)
- **금지**: 검증/테스트 데이터에 대한 **수작업 라벨링·사람 예측** 사용 금지 (§4)
- **순위**: 최종 순위는 **Private Leaderboard** 기준. 동점 시 먼저 제출한 쪽이 우선 (§7)
- **수상자 대응**: 통보 후 **1주 내 응답**, 상금 수령 서류는 **2주 내 제출** (§8, §9)

---

## GENERAL COMPETITION RULES — BINDING AGREEMENT

### 1. ELIGIBILITY

**a.** To be eligible to enter the Competition, you must be:

- a registered account holder at Kaggle.com;
- the older of 18 years old or the age of majority in your jurisdiction of residence (unless otherwise agreed to by Competition Sponsor and appropriate parental/guardian consents have been obtained by Competition Sponsor);
- not a resident of Crimea, so-called Donetsk People's Republic (DNR) or Luhansk People's Republic (LNR), Cuba, Iran, or North Korea; and
- not a person or representative of an entity under U.S. export controls or sanctions (see: <https://www.treasury.gov/resourcecenter/sanctions/Programs/Pages/Programs.aspx>).

**b.** Competitions are open to residents of the United States and worldwide, except that if you are a resident of Crimea, so-called Donetsk People's Republic (DNR) or Luhansk People's Republic (LNR), Cuba, Iran, North Korea, or are subject to U.S. export controls or sanctions, you may not enter the Competition. Other local rules and regulations may apply to you, so please check your local laws to ensure that you are eligible to participate in skills-based competitions. The Competition Host reserves the right to forego or award alternative Prizes where needed to comply with local laws. If a winner is located in a country where prizes cannot be awarded, then they are not eligible to receive a prize.

**c.** If you are entering as a representative of a company, educational institution or other legal entity, or on behalf of your employer, these rules are binding on you, individually, and the entity you represent or where you are an employee. If you are acting within the scope of your employment, or as an agent of another party, you warrant that such party or your employer has full knowledge of your actions and has consented thereto, including your potential receipt of a Prize. You further warrant that your actions do not violate your employer's or entity's policies and procedures.

**d.** The Competition Sponsor reserves the right to verify eligibility and to adjudicate on any dispute at any time. If you provide any false information relating to the Competition concerning your identity, residency, mailing address, telephone number, email address, ownership of right, or information required for entering the Competition, you may be immediately disqualified from the Competition.

### 2. SPONSOR AND HOSTING PLATFORM

**a.** The Competition is sponsored by Competition Sponsor named above. The Competition is hosted on behalf of Competition Sponsor by Kaggle Inc. ("Kaggle"). Kaggle is an independent contractor of Competition Sponsor, and is not a party to this or any agreement between you and Competition Sponsor. You understand that Kaggle has no responsibility with respect to selecting the potential Competition winner(s) or awarding any Prizes. Kaggle will perform certain administrative functions relating to hosting the Competition, and you agree to abide by the provisions relating to Kaggle under these Rules. As a Kaggle.com account holder and user of the Kaggle competition platform, remember you have accepted and are subject to the Kaggle Terms of Service at <www.kaggle.com/terms> in addition to these Rules.

### 3. COMPETITION PERIOD

**a.** For the purposes of Prizes, the Competition will run from the Start Date and time to the Final Submission Deadline (such duration the "Competition Period"). The Competition Timeline is subject to change, and Competition Sponsor may introduce additional hurdle deadlines during the Competition Period. Any updated or additional deadlines will be publicized on the Competition Website. It is your responsibility to check the Competition Website regularly to stay informed of any deadline changes. **YOU ARE RESPONSIBLE FOR DETERMINING THE CORRESPONDING TIME ZONE IN YOUR LOCATION.**

### 4. COMPETITION ENTRY

**a.** **NO PURCHASE NECESSARY TO ENTER OR WIN.** To enter the Competition, you must register on the Competition Website prior to the Entry Deadline, and follow the instructions for developing and entering your Submission through the Competition Website. Your Submissions must be made in the manner and format, and in compliance with all other requirements, stated on the Competition Website (the "Requirements"). Submissions must be received before any Submission deadlines stated on the Competition Website. Submissions not received by the stated deadlines will not be eligible to receive a Prize.

**b.** Submissions may not use or incorporate information from hand labeling or human prediction of the validation dataset or test data records.

**c.** If the Competition is a multi-stage competition with temporally separate training and/or test data, one or more valid Submissions may be required during each Competition stage in the manner described on the Competition Website in order for the Submissions to be Prize eligible.

**d.** Submissions are void if they are in whole or part illegible, incomplete, damaged, altered, counterfeit, obtained through fraud, or late. Competition Sponsor reserves the right to disqualify any entrant who does not follow these Rules, including making a Submission that does not meet the Requirements.

### 5. INDIVIDUALS AND TEAMS

**a. Individual Account.** You may make Submissions only under one, unique Kaggle.com account. You will be disqualified if you make Submissions through more than one Kaggle account, or attempt to falsify an account to act as your proxy. You may submit up to the maximum number of Submissions per day as specified on the Competition Website.

**b. Teams.** If permitted under the Competition Website guidelines, multiple individuals may collaborate as a Team; however, you may join or form only one Team. Each Team member must be a single individual with a separate Kaggle account. You must register individually for the Competition before joining a Team. You must confirm your Team membership to make it official by responding to the Team notification message sent to your Kaggle account. Team membership may not exceed the Maximum Team Size stated on the Competition Website.

**c. Team Merger.** Teams may request to merge via the Competition Website. Team mergers may be allowed provided that:

- (i) the combined Team does not exceed the Maximum Team Size;
- (ii) the number of Submissions made by the merging Teams does not exceed the number of Submissions permissible for one Team at the date of the merger request;
- (iii) the merger is completed before the earlier of: any merger deadline or the Competition deadline; and
- (iv) the proposed combined Team otherwise meets all the requirements of these Rules.

**d. Private Sharing.** No private sharing outside of Teams. Privately sharing code or data outside of Teams is not permitted. It's okay to share code if made available to all Participants on the forums.

### 6. SUBMISSION CODE REQUIREMENTS

**a. Private Code Sharing.** Unless otherwise specifically permitted under the Competition Website or Competition Specific Rules above, during the Competition Period, you are not allowed to privately share source or executable code developed in connection with or based upon the Competition Data or other source or executable code relevant to the Competition ("Competition Code"). This prohibition includes sharing Competition Code between separate Teams, unless a Team merger occurs. Any such sharing of Competition Code is a breach of these Competition Rules and may result in disqualification.

**b. Public Code Sharing.** You are permitted to publicly share Competition Code, provided that such public sharing does not violate the intellectual property rights of any third party. If you do choose to share Competition Code or other such code, you are required to share it on Kaggle.com on the discussion forum or notebooks associated specifically with the Competition for the benefit of all competitors. By so sharing, you are deemed to have licensed the shared code under an Open Source Initiative-approved license (see <www.opensource.org>) that in no event limits commercial use of such Competition Code or model containing or depending on such Competition Code.

**c. Use of Open Source.** Unless otherwise stated in the Specific Competition Rules above, if open source code is used in the model to generate the Submission, then you must only use open source code licensed under an Open Source Initiative-approved license (see <www.opensource.org>) that in no event limits commercial use of such code or model containing or depending on such code.

### 7. DETERMINING WINNERS

**a.** Each Submission will be scored and ranked by the evaluation metric stated on the Competition Website. During the Competition Period, the current ranking will be visible on the Competition Website's Public Leaderboard. The potential winner(s) are determined solely by the leaderboard ranking on the Private Leaderboard, subject to compliance with these Rules. The Public Leaderboard will be based on the public test set and the Private Leaderboard will be based on the private test set.

**b.** In the event of a tie, the Submission that was entered first to the Competition will be the winner. In the event a potential winner is disqualified for any reason, the Submission that received the next highest score rank will be chosen as the potential winner.

### 8. NOTIFICATION OF WINNERS & DISQUALIFICATION

**a.** The potential winner(s) will be notified by email.

**b.** If a potential winner (i) does not respond to the notification attempt within one (1) week from the first notification attempt or (ii) notifies Kaggle within one week after the Final Submission Deadline that the potential winner does not want to be nominated as a winner or does not want to receive a Prize, then, in each case (i) and (ii) such potential winner will not receive any Prize, and an alternate potential winner will be selected from among all eligible entries received based on the Competition's judging criteria.

**c.** In case (i) and (ii) above Kaggle may disqualify the Participant. However, in case (ii) above, if requested by Kaggle, such potential winner may provide code and documentation to verify the Participant's compliance with these Rules. If the potential winner provides code and documentation to the satisfaction of Kaggle, the Participant will not be disqualified pursuant to this paragraph.

**d.** Competition Sponsor reserves the right to disqualify any Participant from the Competition if the Competition Sponsor reasonably believes that the Participant has attempted to undermine the legitimate operation of the Competition by cheating, deception, or other unfair playing practices or abuses, threatens or harasses any other Participants, Competition Sponsor or Kaggle.

**e.** A disqualified Participant may be removed from the Competition leaderboard, at Kaggle's sole discretion. If a Participant is removed from the Competition Leaderboard, additional winning features associated with the Kaggle competition platform, for example Kaggle points or medals, may also not be awarded.

**f.** The final leaderboard list will be publicly displayed at Kaggle.com. Determinations of Competition Sponsor are final and binding.

### 9. PRIZES

**a.** Prize(s) are as described on the Competition Website and are only available for winning during the time period described on the Competition Website. The odds of winning any Prize depends on the number of eligible Submissions received during the Competition Period and the skill of the Participants.

**b.** All Prizes are subject to Competition Sponsor's review and verification of the Participant's eligibility and compliance with these Rules, and the compliance of the winning Submissions with the Submissions Requirements. In the event that the Submission demonstrates non-compliance with these Competition Rules, Competition Sponsor may at its discretion take either of the following actions: (i) disqualify the Submission(s); or (ii) require the potential winner to remediate within one week after notice all issues identified in the Submission(s) (including, without limitation, the resolution of license conflicts, the fulfillment of all obligations required by software licenses, and the removal of any software that violates the software restrictions).

**c.** A potential winner may decline to be nominated as a Competition winner in accordance with Section 3.8.

**d.** Potential winners must return all required Prize acceptance documents within two (2) weeks following notification of such required documents, or such potential winner will be deemed to have forfeited the prize and another potential winner will be selected. Prize(s) will be awarded within approximately thirty (30) days after receipt by Competition Sponsor or Kaggle of the required Prize acceptance documents. Transfer or assignment of a Prize is not allowed.

**e.** You are not eligible to receive any Prize if you do not meet the Eligibility requirements in Section 2.7 and Section 3.1 above.

**f.** If a Team wins a monetary Prize, the Prize money will be allocated in even shares between the eligible Team members, unless the Team unanimously opts for a different Prize split and notifies Kaggle before Prizes are issued.

### 10. TAXES

**a.** **ALL TAXES IMPOSED ON PRIZES ARE THE SOLE RESPONSIBILITY OF THE WINNERS.** Payments to potential winners are subject to the express requirement that they submit all documentation requested by Competition Sponsor or Kaggle for compliance with applicable state, federal, local and foreign (including provincial) tax reporting and withholding requirements. Prizes will be net of any taxes that Competition Sponsor is required by law to withhold. If a potential winner fails to provide any required documentation or comply with applicable laws, the Prize may be forfeited and Competition Sponsor may select an alternative potential winner. Any winners who are U.S. residents will receive an IRS Form-1099 in the amount of their Prize.

### 11. GENERAL CONDITIONS

**a.** All federal, state, provincial and local laws and regulations apply.

### 12. PUBLICITY

**a.** You agree that Competition Sponsor, Kaggle and its affiliates may use your name and likeness for advertising and promotional purposes without additional compensation, unless prohibited by law.

### 13. PRIVACY

**a.** You acknowledge and agree that Competition Sponsor and Kaggle may collect, store, share and otherwise use personally identifiable information provided by you during the Kaggle account registration process and the Competition, including but not limited to, name, mailing address, phone number, and email address ("Personal Information"). Kaggle acts as an independent controller with regard to its collection, storage, sharing, and other use of this Personal Information, and will use this Personal Information in accordance with its Privacy Policy <www.kaggle.com/privacy>, including for administering the Competition. As a Kaggle.com account holder, you have the right to request access to, review, rectification, portability or deletion of any personal data held by Kaggle about you by logging into your account and/or contacting Kaggle Support at <www.kaggle.com/contact>.

**b.** As part of Competition Sponsor performing this contract between you and the Competition Sponsor, Kaggle will transfer your Personal Information to Competition Sponsor, which acts as an independent controller with regard to this Personal Information. As a controller of such Personal Information, Competition Sponsor agrees to comply with all U.S. and foreign data protection obligations with regard to your Personal Information. Kaggle will transfer your Personal Information to Competition Sponsor in the country specified in the Competition Sponsor Address listed above, which may be a country outside the country of your residence. Such country may not have privacy laws and regulations similar to those of the country of your residence.

### 14. WARRANTY, INDEMNITY AND RELEASE

**a.** You warrant that your Submission is your own original work and, as such, you are the sole and exclusive owner and rights holder of the Submission, and you have the right to make the Submission and grant all required licenses. You agree not to make any Submission that: (i) infringes any third party proprietary rights, intellectual property rights, industrial property rights, personal or moral rights or any other rights, including without limitation, copyright, trademark, patent, trade secret, privacy, publicity or confidentiality obligations, or defames any person; or (ii) otherwise violates any applicable U.S. or foreign state or federal law.

**b.** To the maximum extent permitted by law, you indemnify and agree to keep indemnified Competition Entities at all times from and against any liability, claims, demands, losses, damages, costs and expenses resulting from any of your acts, defaults or omissions and/or a breach of any warranty set forth herein. To the maximum extent permitted by law, you agree to defend, indemnify and hold harmless the Competition Entities from and against any and all claims, actions, suits or proceedings, as well as any and all losses, liabilities, damages, costs and expenses (including reasonable attorneys fees) arising out of or accruing from:

- (a) your Submission or other material uploaded or otherwise provided by you that infringes any third party proprietary rights, intellectual property rights, industrial property rights, personal or moral rights or any other rights, including without limitation, copyright, trademark, patent, trade secret, privacy, publicity or confidentiality obligations, or defames any person;
- (b) any misrepresentation made by you in connection with the Competition;
- (c) any non-compliance by you with these Rules or any applicable U.S. or foreign state or federal law;
- (d) claims brought by persons or entities other than the parties to these Rules arising from or related to your involvement with the Competition; and
- (e) your acceptance, possession, misuse or use of any Prize, or your participation in the Competition and any Competition-related activity.

**c.** You hereby release Competition Entities from any liability associated with: (a) any malfunction or other problem with the Competition Website; (b) any error in the collection, processing, or retention of any Submission; or (c) any typographical or other error in the printing, offering or announcement of any Prize or winners.

### 15. INTERNET

**a.** Competition Entities are not responsible for any malfunction of the Competition Website or any late, lost, damaged, misdirected, incomplete, illegible, undeliverable, or destroyed Submissions or entry materials due to system errors, failed, incomplete or garbled computer or other telecommunication transmission malfunctions, hardware or software failures of any kind, lost or unavailable network connections, typographical or system/human errors and failures, technical malfunction(s) of any telephone network or lines, cable connections, satellite transmissions, servers or providers, or computer equipment, traffic congestion on the Internet or at the Competition Website, or any combination thereof, which may limit a Participant's ability to participate.

### 16. RIGHT TO CANCEL, MODIFY OR DISQUALIFY

**a.** If for any reason the Competition is not capable of running as planned, including infection by computer virus, bugs, tampering, unauthorized intervention, fraud, technical failures, or any other causes which corrupt or affect the administration, security, fairness, integrity, or proper conduct of the Competition, Competition Sponsor reserves the right to cancel, terminate, modify or suspend the Competition. Competition Sponsor further reserves the right to disqualify any Participant who tampers with the submission process or any other part of the Competition or Competition Website. Any attempt by a Participant to deliberately damage any website, including the Competition Website, or undermine the legitimate operation of the Competition is a violation of criminal and civil laws. Should such an attempt be made, Competition Sponsor and Kaggle each reserves the right to seek damages from any such Participant to the fullest extent of the applicable law.

### 17. NOT AN OFFER OR CONTRACT OF EMPLOYMENT

**a.** Under no circumstances will the entry of a Submission, the awarding of a Prize, or anything in these Rules be construed as an offer or contract of employment with Competition Sponsor or any of the Competition Entities. You acknowledge that you have submitted your Submission voluntarily and not in confidence or in trust. You acknowledge that no confidential, fiduciary, agency, employment or other similar relationship is created between you and Competition Sponsor or any of the Competition Entities by your acceptance of these Rules or your entry of your Submission.

### 18. DEFINITIONS

| 용어 | 정의 |
|------|------|
| **a. Competition Data** | The data or datasets available from the Competition Website for the purpose of use in the Competition, including any prototype or executable code provided on the Competition Website. The Competition Data will contain private and public test sets. Which data belongs to which set will not be made available to Participants. |
| **b. Entry** | When a Participant has joined, signed up, or accepted the rules of a competition. Entry is required to make a Submission to a competition. |
| **c. Final Submission** | The Submission selected by the user, or automatically selected by Kaggle in the event not selected by the user, that is/are used for final placement on the competition leaderboard. |
| **d. Participant / Participant User** | An individual who participates in a competition by entering the competition and making a Submission. |
| **e. Private Leaderboard** | A ranked display of Participants' Submission scores against the private test set. The Private Leaderboard determines the final standing in the competition. |
| **f. Public Leaderboard** | A ranked display of Participants' Submission scores against a representative sample of the test data. This leaderboard is visible throughout the competition. |
| **g. Sponsor** | Responsible for hosting the competition, which includes but is not limited to providing the data for the competition, determining winners, and enforcing competition rules. |
| **h. Submission** | Anything provided by the Participant to the Sponsor to be evaluated for competition purposes and determine leaderboard position. A Submission may be made as a model, notebook, prediction file, or other format as determined by the Sponsor. |
| **i. Team** | One or more Participants participating together in a Kaggle competition, by officially merging together as a Team within the competition platform. |
