#ifndef __ARGS_PARSER_H_
#define __ARGS_PARSER_H_


#include <unordered_map>
#include <string>
#include <regex>
#include <iomanip>
#include <iostream>

namespace argsParser
{
    using std::string;

    enum class P_DATA_TYPE
    {
        PARSER_BOOL,
        PARSER_INT,
        PARSER_FLOAT,
        PARSER_DOUBLE,
        PARSER_STRING
    };

    struct parserInfo
    {
        string desc;
        string defaultValue;
        string valueDesc;

        P_DATA_TYPE dataType;
        string value;
    };

    typedef string Desc;
    typedef string ValueDesc;
    typedef string DefaultValue;

    class parser
    {

        #define ADD_ARG_FUNCS(DATA_TYPE) \
        static void ADD_ARG_##DATA_TYPE(string name,Desc desc,DefaultValue defaultValue,ValueDesc valueDesc =""){   \
            InnerInitArgs(name,desc,defaultValue,valueDesc,P_DATA_TYPE::PARSER_##DATA_TYPE); \
        }

        public:
        static void InnerInitArgs(string name,Desc desc,DefaultValue defaultValue,ValueDesc valueDesc,P_DATA_TYPE dataType)
        {
            mArgs.emplace(std::make_pair(name, parserInfo{desc,defaultValue,valueDesc, dataType ,defaultValue}));
        }

        ADD_ARG_FUNCS(INT);
        ADD_ARG_FUNCS(FLOAT);
        ADD_ARG_FUNCS(DOUBLE);
        ADD_ARG_FUNCS(STRING);
        ADD_ARG_FUNCS(BOOL);

        static void printDesc()
        {   
            for (const auto& data :mArgs )
            {
                string name = data.first;
                auto& info = data.second;
                
                if(info.valueDesc.length() > 0)
                    name += "=<" + info.valueDesc + ">"; 

                std::cout << std::left << std::setw(20) << name;
                std::cout << std::setw(2) << "=" << std::setw(2);
                std::cout << std::left << std::setw(80) << info.desc + "(default:" + info.defaultValue + ")";
                std::cout << std::endl;
            }
        }

        static void parseArgs(int argc,char** argv)
        {
            string* str_argvs = new string[argc];
            for(int i = 0;i<argc;++i){
                str_argvs[i] = argv[i];
                //std::cout << argv[i] << " " << std::endl;
            }

            std::regex args_regex(R"(--(.+)=(.+))");
            std::smatch matches;
            for (int i = 1;i<argc;++i) {
                if(std::regex_match(str_argvs[i], matches, args_regex) && matches.size() ==3 )
                {   
                    string key = matches[1].str();
                    string value = matches[2].str();
                    if (mArgs.find(key)!=mArgs.end())
                        mArgs[key].value = value;
                    else
                        std::cout << "do not have the param named:" << key << " ";
                }
                else
                    std::cout << "set param wrong ,need \'--{param}={value}\'" << std::endl;
            } 

            if(str_argvs)
                delete [] str_argvs;

            std::cout << "####### input args####### " << std::endl;
            for (const auto& data :mArgs )
                std::cout << data.first << "=" << data.second.value << ";\n" ;
            std::cout << "####### end args####### " << std::endl;

        }

        static int getIntValue(string name)
        {
            return mArgs.find(name)!=mArgs.end() ? std::stoi( mArgs[name].value) : 0;
        }

        static float getFloatValue(string name)
        {
            return mArgs.find(name)!=mArgs.end() ? std::stof( mArgs[name].value) : 0.0f;
        }

        static double getDoubleValue(string name)
        {
            return mArgs.find(name)!=mArgs.end() ? std::stod( mArgs[name].value) != 0 : 0.0;
        }

        static string getStringValue(string name)
        {
            return mArgs.find(name)!=mArgs.end() ? mArgs[name].value : "";
        }

        static bool getBoolValue(string name)
        {
            return mArgs.find(name)!=mArgs.end() ? std::stoi( mArgs[name].value) != 0 : 0;
        }


    private:
        static std::map<string,parserInfo> mArgs;
    };

    std::map<string,parserInfo> parser::mArgs ;
};


#endif
