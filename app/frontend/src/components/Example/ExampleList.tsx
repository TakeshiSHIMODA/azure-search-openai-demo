import { Example } from "./Example";

import styles from "./Example.module.css";

export type ExampleModel = {
    text: string;
    value: string;
};

const EXAMPLES: ExampleModel[] = [
    {
        text: "ハイブリッド車の高出力化の課題は？",
        value: "ハイブリッド車の高出力化の課題は？"
    },
    { 
        text: "BEVの開発にはどのような課題がありますか", 
        value: "BEVの開発にはどのような課題がありますか" 
    },
    { 
        text: "デュアルインバーターシステムの特徴を教えてください", 
        value: "デュアルインバーターシステムの特徴を教えてください" 
    }
];

interface Props {
    onExampleClicked: (value: string) => void;
}

export const ExampleList = ({ onExampleClicked }: Props) => {
    return (
        <ul className={styles.examplesNavList}>
            {EXAMPLES.map((x, i) => (
                <li key={i}>
                    <Example text={x.text} value={x.value} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
