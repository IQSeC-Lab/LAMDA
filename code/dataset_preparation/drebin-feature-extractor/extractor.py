import os
import re
import json
import subprocess
import logging
import multiprocessing
import argparse
import tldextract
import shutil
from xml.dom import minidom
from typing import List, Dict, Set, Optional

# Setup logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("runtime.log"),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)

# File to store failed APKs
FAILED_APKS_FILE = "./failed_apks.txt"

class PScoutMapping:
    def __init__(self):
        try:
            # Load permission-API mapping from JSON
            with open('./PScount/SmallCasePScoutPermApiDict.json', 'r') as FH:
                self.PermApiDictFromJsonTemp = json.load(FH)
                self.PermApiDictFromJson = {}
                for Perms in self.PermApiDictFromJsonTemp:
                    for Api in self.PermApiDictFromJsonTemp[Perms]:
                        ApiName = Api[0].lower() + Api[1].lower()
                        self.PermApiDictFromJson[ApiName] = Perms
            del self.PermApiDictFromJsonTemp
        except Exception as e:
            logger.error(f"Failed to load PScout mapping: {e}")
            raise

    def GetPermFromApi(self, ApiClass: str, ApiMethodName: str) -> Optional[str]:
        ApiName = ApiClass.lower() + ApiMethodName.lower()
        return self.PermApiDictFromJson.get(ApiName, None)

class SmaliFeatures:
    def __init__(self):
        self.network_address: Set[str] = set()
        self.used_permission: Set[str] = set()
        self.suspicious_apicall: Set[str] = set()
        self.restricted_apicall: Set[str] = set()
        self.required_permissions: Set[str] = set()
        self.activities: Set[str] = set()
        self.services: Set[str] = set()
        self.content_providers: Set[str] = set()
        self.broadcast_receivers: Set[str] = set()
        self.hardware_components: Set[str] = set()
        self.intent_filters: Set[str] = set()
        self.PMap = PScoutMapping()

    def find_feature(self, path: str) -> tuple:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            network_address = self.find_network_feature(lines)
            api_call, suspicious_apicall = self.find_invoked_Android_APIs(lines)
            used_permission, restricted_apicall = self.get_permissions_and_API(api_call, self.PMap, self.required_permissions)

            return network_address, used_permission, suspicious_apicall, restricted_apicall
        except Exception as e:
            logger.error(f"Error processing file {path}: {e}")
            return [], set(), set(), set()

    #     return list(URLDomainSet)
    def find_network_feature(self, instructions: List[str]) -> List[str]:
        URLDomainSet = set()
        ip_pattern = r'^((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.?\b){4}$'  # Matches IPv4 addresses
        # url_pattern = r'https?://([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})(?:[/:][^\s]*)?'  # fail to diff url ending with .
        url_pattern = r'https?://([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})(?=[/:\s]|$)'
        for instruction in instructions:
            instruction = instruction.strip()

            # Extract IPs
            ip_match = re.findall(ip_pattern, instruction)
            if ip_match:
                URLDomainSet.update(ip_match)
            # Extract valid URLs & domains
            url_match = re.findall(url_pattern, instruction)
            for domain in url_match:
                extracted = tldextract.extract(domain)

                # Ensure we ignore malformed domains like "books.google."
                if extracted.suffix:  # Only process if a valid TLD is found
                    if extracted.subdomain:  
                        full_domain = f"{extracted.subdomain}.{extracted.domain}.{extracted.suffix}"
                    else:
                        full_domain = f"{extracted.domain}.{extracted.suffix}"
                    
                    URLDomainSet.add(full_domain)

        return list(URLDomainSet)

    def get_permissions_and_API(self, ApiList: List[Dict], PMap: PScoutMapping, RequestedPermissionList: Set[str]) -> tuple:
        PermissionSet = set()
        RestrictedApiSet = set()

        for Api in ApiList:
            ApiClass = Api['ApiClass'].replace("/", ".").replace("Landroid", "android").strip()
            Permission = PMap.GetPermFromApi(ApiClass, Api['ApiName'])
            if Permission:
                if Permission in RequestedPermissionList:
                    PermissionSet.add(Permission)
                else:
                    RestrictedApiSet.add(f"{ApiClass}.{Api['ApiName']}")

        return PermissionSet, RestrictedApiSet

    def find_invoked_Android_APIs(self, DalvikCodeList: List[str]) -> tuple:
        ApiList = []
        SuspiciousApiSet = set()

        AndroidSuspiciousApiNameList = {"getExternalStorageDirectory", "getSimCountryIso", "execHttpRequest", 
                "sendTextMessage", "getSubscriberId", "getDeviceId", "getPackageInfo", "getSystemService", "getWifiState", 
                "setWifiEnabled", "setWifiDisabled", "Cipher"}
        OtherSuspiciousApiNameList = {"Ljava/net/HttpURLconnection;->setRequestMethod(Ljava/lang/String;)", "Ljava/net/HttpURLconnection", 
                                  "Lorg/apache/http/client/methods/HttpPost", "Landroid/telephony/SmsMessage;->getMessageBody", 
                                  "Ljava/io/IOException;->printStackTrace", "Ljava/lang/Runtime;->exec"}
        NotLikeApiNameList = {"system/bin/su", "android/os/Exec"}

        for DalvikCode in DalvikCodeList:
            if "invoke-" in DalvikCode:
                parts = DalvikCode.split(",")
                for part in parts:
                    if ";->" in part:
                        part = part.strip()
                        if part.startswith('Landroid'):
                            ApiClass, ApiName = part.split(";->")[0], part.split(";->")[1].split("(")[0]
                            ApiList.append({'FullApi': part, 'ApiClass': ApiClass, 'ApiName': ApiName})
                            if ApiName in AndroidSuspiciousApiNameList:
                                SuspiciousApiSet.add(f"{ApiClass}.{ApiName}")
                    for element in OtherSuspiciousApiNameList:
                        if element in part:
                            SuspiciousApiSet.add(element)
            for element in NotLikeApiNameList:
                if element in DalvikCode:
                    SuspiciousApiSet.add(element)

        return ApiList, SuspiciousApiSet

    def listDir(self, rootDir: str):
        for root, _, files in os.walk(rootDir):
            for file in files:
                if file.endswith(".smali"):
                    path = os.path.join(root, file)
                    tmp_network_address, tmp_used_permission, tmp_suspicious_apicall, tmp_restricted_apicall = self.find_feature(path)
                    self.network_address.update(tmp_network_address)
                    self.used_permission.update(tmp_used_permission)
                    self.suspicious_apicall.update(tmp_suspicious_apicall)
                    self.restricted_apicall.update(tmp_restricted_apicall)

    def parse_manifest(self, manifest_path: str):
        """Parse AndroidManifest.xml to extract additional information"""
        try:
            dom = minidom.parse(manifest_path)
            dom_collection = dom.documentElement

            # Extract permissions
            for permission in dom_collection.getElementsByTagName("uses-permission"):
                if permission.hasAttribute("android:name"):
                    self.required_permissions.add(permission.getAttribute("android:name"))

            # Extract activities
            for activity in dom_collection.getElementsByTagName("activity"):
                if activity.hasAttribute("android:name"):
                    self.activities.add(activity.getAttribute("android:name"))

            # Extract services
            for service in dom_collection.getElementsByTagName("service"):
                if service.hasAttribute("android:name"):
                    self.services.add(service.getAttribute("android:name"))

            # Extract content providers
            for provider in dom_collection.getElementsByTagName("provider"):
                if provider.hasAttribute("android:name"):
                    self.content_providers.add(provider.getAttribute("android:name"))

            # Extract broadcast receivers
            for receiver in dom_collection.getElementsByTagName("receiver"):
                if receiver.hasAttribute("android:name"):
                    self.broadcast_receivers.add(receiver.getAttribute("android:name"))

            # Extract hardware components
            for hardware in dom_collection.getElementsByTagName("uses-feature"):
                if hardware.hasAttribute("android:name"):
                    self.hardware_components.add(hardware.getAttribute("android:name"))

            # Extract intent filters
            for intent_filter in dom_collection.getElementsByTagName("intent-filter"):
                for action in intent_filter.getElementsByTagName("action"):
                    if action.hasAttribute("android:name"):
                        self.intent_filters.add(action.getAttribute("android:name"))

        except Exception as e:
            logger.error(f"Failed to parse AndroidManifest.xml: {e}")

def decompile_apk(apk_path: str, output_dir: str, apktool_jar: str) -> Optional[str]:
    """Decompile APK using apktool"""
    # here output_dir parse the original apk filename which is defined in extract_features_from_apk func
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cmd = f"java -jar {apktool_jar} d {apk_path} -o {output_dir} --force -q"
        subprocess.run(cmd, shell=True, check=True)
        return output_dir
    except Exception as e:
        logger.error(f"Failed to decompile APK {apk_path}: {e}")
        return None

def extract_features_from_apk(apk_path: str, apktool_jar: str, result_dir: str) -> Optional[List[str]]:
    """Extract features from an APK file and pick any two from below either ram based or I/O based"""
    output_dir = decompile_apk(apk_path, os.path.join(output_dir, os.path.splitext(os.path.basename(apk_path))[0]), apktool_jar) # use this for I/O based decompilatiion | slow processing
    #output_dir = decompile_apk(apk_path, f"/mnt/ramdisk/{os.path.splitext(os.path.basename(apk_path))[0]}", apktool_jar) # use this for ram based decompilatiion for fast processing

    if not output_dir:
        return None

    SF = SmaliFeatures()
    AndroidManifest = os.path.join(output_dir, "AndroidManifest.xml")
    # print(AndroidManifest)
    # Copy AndroidManifest.xml to the output directory
    if os.path.exists(AndroidManifest):
        shutil.copy(AndroidManifest, os.path.join(result_dir, f"{os.path.splitext(os.path.basename(apk_path))[0]}.xml"))
    #    print(os.path.join(output_dir, f"{os.path.splitext(os.path.basename(apk_path))[0]}.xml"))
    # Parse AndroidManifest.xml
    if os.path.exists(AndroidManifest):
        SF.parse_manifest(AndroidManifest)

    SF.listDir(output_dir)

    # Format the output
    output = []

    # Requested Permissions
    for permission in SF.required_permissions:
        output.append(f"RequestedPermissionList_{permission}")

    # Activities
    for activity in SF.activities:
        output.append(f"ActivityList_{activity}")

    # Services
    for service in SF.services:
        output.append(f"ServiceList_{service}")

    # Broadcast Receivers
    for receiver in SF.broadcast_receivers:
        output.append(f"BroadcastReceiverList_{receiver}")

    # Hardware Components
    for hardware in SF.hardware_components:
        output.append(f"HardwareComponentsList_{hardware}")

    # Intent Filters
    for intent_filter in SF.intent_filters:
        output.append(f"IntentFilterList_{intent_filter}")

    # Used Permissions
    for permission in SF.used_permission:
        output.append(f"UsedPermissionsList_{permission}")

    # Restricted API Calls
    for api_call in SF.restricted_apicall:
        output.append(f"RestrictedApiList_{api_call}")

    # Suspicious API Calls
    for api_call in SF.suspicious_apicall:
        output.append(f"SuspiciousApiList_{api_call}")

    for url in SF.network_address:
        output.append(f"URLDomainList_{url}")
    # Cleanup: Remove the decompiled directory
    try:
        shutil.rmtree(output_dir)
        logger.info(f"Removed decompiled directory: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to remove decompiled directory {output_dir}: {e}")
    return output

def process_apk(apk_path: str, apktool_jar: str, result_dir: str) -> Optional[str]:
    """Wrapper function to process a single APK."""
    try:
        logger.info(f"Processing APK: {apk_path}")
        features = extract_features_from_apk(apk_path, apktool_jar, result_dir)
        if features:
            # Save features to a .data file
            output_file = os.path.join(result_dir, f"{os.path.splitext(os.path.basename(apk_path))[0]}.data")
            with open(output_file, 'w') as f:
                for line in features:
                    f.write(f"{line}\n")
            logger.info(f"Saved features for {apk_path} to {output_file}")
            return output_file
        else:
            logger.warning(f"Failed to process APK: {apk_path}")
            return None
    except Exception as e:
        logger.error(f"Error processing APK {apk_path}: {e}")
        return None

def process_apks_in_parallel(apk_dir: str, apktool_jar: str, result_dir: str):
    """Process all APKs in a directory in parallel."""
    # apk_files = [os.path.join(apk_dir, f) for f in os.listdir(apk_dir) if f.endswith(".apk")]
    apk_files = [
        os.path.join(apk_dir, f)
        for f in os.listdir(apk_dir)
        if f.endswith(".apk") and not os.path.exists(os.path.join(result_dir, os.path.splitext(f)[0] + ".data"))
    ]
    # print(apk_files)
    if not apk_files:
        logger.warning(f"No new APK files found in directory: {apk_dir}")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # apk_files = [apk for apk in apk_files if not os.path.exists(os.path.join(result_dir, os.path.splitext(os.path.basename(apk))[0]))]

    if not apk_files:
        logger.info("All APKs have already been processed.")
        return
    # Process APKs in parallel
    with multiprocessing.Pool(processes=10) as pool: # Adjust the number of processes as needed
        results = pool.starmap(process_apk, [(apk, apktool_jar, result_dir) for apk in apk_files])

    # Track failed APKs
    failed_apks = [os.path.splitext(os.path.basename(apk_files[i]))[0] for i, result in enumerate(results) if result is None]

    # Save failed APKs to a file
    if failed_apks:
        with open(os.path.join(FAILED_APKS_FILE), 'w') as f:
            for apk in failed_apks:
                f.write(f"{apk}\n")
        logger.info(f"Saved list of failed APKs to {os.path.join(FAILED_APKS_FILE)}")

if __name__ == "__main__":
    # os.system('touch runtime.log') #linux only
    parser = argparse.ArgumentParser(description="Process APKs to extract features.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing APKs")
    parser.add_argument("--result_dir", type=str, required=True, help="Directory to save feature outputs as .data")
    parser.add_argument("--apktool", type=str, default="./apktool.jar", help="Path to apktool.jar")
    args = parser.parse_args()

    process_apks_in_parallel(args.input_dir, args.apktool, args.result_dir)